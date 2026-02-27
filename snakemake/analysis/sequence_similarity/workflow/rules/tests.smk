"""Synthetic data tests for MMseqs2 repeat masking behaviour.

Generates synthetic 256bp sequences with soft-masked (lowercase) repeats and
verifies that MMseqs2 --mask-lower-case correctly excludes repeat regions from
k-mer matching for both ``mmseqs cluster`` and ``mmseqs search``.

Note: ``mmseqs linclust`` silently ignores ``--mask-lower-case``, so only
``mmseqs cluster`` and ``mmseqs search`` are tested here.
"""

import random
from textwrap import fill


# =============================================================================
# Constants
# =============================================================================

_TEST_SEQ_LEN = 256
_LONG_REPEAT_LEN = 200  # bases of repeat in repeat_only sequences
_SHORT_REPEAT_LEN = 50  # bases of repeat in mixed sequences
_MUTATION_RATE = 0.10  # ~10% point mutations for "genuine_B"
_FASTA_LINE_WIDTH = 80
_SEED = 42


# =============================================================================
# Helpers (only used within this file)
# =============================================================================


def _random_dna(length: int, rng: random.Random) -> str:
    """Return a random uppercase DNA string of the given length."""
    return "".join(rng.choice("ACGT") for _ in range(length))


def _random_repeat_block(length: int, rng: random.Random) -> str:
    """Return a random *high-complexity* lowercase block simulating a TE fragment.

    A simple dinucleotide repeat (e.g. ``atatat…``) has only two unique
    k-mers and may be caught by MMseqs2's built-in compositional bias
    filter regardless of ``--mask-lower-case``.  Using a random sequence
    ensures the test isolates case-based masking from low-complexity
    masking.
    """
    return "".join(rng.choice("acgt") for _ in range(length))


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    """Introduce point mutations at the given rate, preserving case."""
    bases = list(seq)
    for i in range(len(bases)):
        if rng.random() < rate:
            original = bases[i].upper()
            choices = [b for b in "ACGT" if b != original]
            replacement = rng.choice(choices)
            # Preserve case: if original was lowercase, keep lowercase
            bases[i] = replacement.lower() if bases[i].islower() else replacement
    return "".join(bases)


def _write_fasta(path: str, records: list[tuple[str, str]]) -> None:
    """Write a list of (name, sequence) tuples to a FASTA file."""
    with open(path, "w") as f:
        for name, seq in records:
            f.write(f">{name}\n")
            f.write(fill(seq, width=_FASTA_LINE_WIDTH) + "\n")


# =============================================================================
# Rules
# =============================================================================


rule generate_synthetic_test_data:
    """Generate synthetic FASTA with soft-masked repeat sequences."""
    output:
        fasta="results/tests/synthetic.fasta",
    run:
        rng = random.Random(_SEED)

        # --- genuine pair: random non-repetitive DNA -------------------------
        genuine_A = _random_dna(_TEST_SEQ_LEN, rng)
        genuine_B = _mutate(genuine_A, _MUTATION_RATE, rng)

        # --- repeat-only pair: different flanks, same lowercase repeat --------
        # Use a high-complexity random repeat (simulates a TE fragment)
        repeat_block = _random_repeat_block(_LONG_REPEAT_LEN, rng)
        flank_len = (_TEST_SEQ_LEN - _LONG_REPEAT_LEN) // 2
        tail_len = _TEST_SEQ_LEN - _LONG_REPEAT_LEN - flank_len

        flank_A_left = _random_dna(flank_len, rng)
        flank_A_right = _random_dna(tail_len, rng)
        repeat_only_A = flank_A_left + repeat_block + flank_A_right

        flank_B_left = _random_dna(flank_len, rng)
        flank_B_right = _random_dna(tail_len, rng)
        repeat_only_B = flank_B_left + repeat_block + flank_B_right

        # --- mixed pair: genuine sequence + short lowercase repeat insert -----
        insert_pos = _TEST_SEQ_LEN // 2
        short_repeat = repeat_block[:_SHORT_REPEAT_LEN]

        mixed_A_base = genuine_A[:insert_pos] + short_repeat + genuine_A[insert_pos:]
        mixed_A = mixed_A_base[:_TEST_SEQ_LEN]  # trim to 256bp

        mixed_B_base = genuine_B[:insert_pos] + short_repeat + genuine_B[insert_pos:]
        mixed_B = mixed_B_base[:_TEST_SEQ_LEN]  # trim to 256bp

        records = [
            ("genuine_A", genuine_A),
            ("genuine_B", genuine_B),
            ("repeat_only_A", repeat_only_A),
            ("repeat_only_B", repeat_only_B),
            ("mixed_A", mixed_A),
            ("mixed_B", mixed_B),
        ]

        _write_fasta(str(output.fasta), records)

        # Log sequence composition
        for name, seq in records:
            n_lower = sum(1 for c in seq if c.islower())
            print(f"  {name}: {len(seq)}bp, {n_lower} lowercase ({100*n_lower/len(seq):.0f}%)")


rule run_mmseqs_mask_test:
    """Run MMseqs2 cluster on synthetic data with mask-lower-case={mask_lower_case}.

    Uses ``mmseqs cluster`` (not ``linclust``) because linclust silently
    ignores ``--mask-lower-case``.
    """
    input:
        fasta="results/tests/synthetic.fasta",
    output:
        tsv="results/tests/clusters_masklc{mask_lower_case}.tsv",
    params:
        db="results/tests/mmseqs_masklc{mask_lower_case}/db",
        cluster_prefix="results/tests/mmseqs_masklc{mask_lower_case}/clusters",
        tmp="results/tests/mmseqs_masklc{mask_lower_case}/tmp",
    wildcard_constraints:
        mask_lower_case="[01]",
    threads: 1
    resources:
        mem_mb=512,
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p $(dirname {params.db})
        mmseqs createdb {input.fasta} {params.db} \
            --mask-lower-case {wildcards.mask_lower_case} \
            --threads {threads}
        mmseqs cluster {params.db} {params.cluster_prefix} {params.tmp} \
            --mask-lower-case {wildcards.mask_lower_case} \
            --min-seq-id 0.5 \
            -c 0.5 \
            --cov-mode 0 \
            --threads {threads}
        mmseqs createtsv {params.db} {params.db} {params.cluster_prefix} {output.tsv} \
            --threads {threads}
        """


rule verify_masking_results:
    """Compare MMseqs2 clustering with and without --mask-lower-case."""
    input:
        mask0="results/tests/clusters_masklc0.tsv",
        mask1="results/tests/clusters_masklc1.tsv",
    output:
        summary="results/tests/masking_test_summary.txt",
    run:
        def parse_clusters(path):
            """Return dict mapping each sequence to its cluster representative."""
            seq_to_rep = {}
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        rep, member = parts[0], parts[1]
                        seq_to_rep[member] = rep
            return seq_to_rep

        def same_cluster(clusters, a, b):
            return clusters.get(a) == clusters.get(b)

        clusters_mask0 = parse_clusters(str(input.mask0))
        clusters_mask1 = parse_clusters(str(input.mask1))

        # Expected results:
        # - genuine pair: similar by real homology → same cluster in BOTH modes
        # - repeat_only pair: similar ONLY via repeat → same cluster without mask,
        #   DIFFERENT clusters with mask
        # - mixed pair: genuine homology + repeat → same cluster in BOTH modes
        expectations = [
            # (description, seq_a, seq_b, expected_mask0, expected_mask1)
            ("genuine_A ↔ genuine_B (mask=0: same)",
             "genuine_A", "genuine_B", True, None),
            ("genuine_A ↔ genuine_B (mask=1: same)",
             "genuine_A", "genuine_B", None, True),
            ("repeat_only_A ↔ repeat_only_B (mask=0: same)",
             "repeat_only_A", "repeat_only_B", True, None),
            ("repeat_only_A ↔ repeat_only_B (mask=1: different)",
             "repeat_only_A", "repeat_only_B", None, False),
            ("mixed_A ↔ mixed_B (mask=0: same)",
             "mixed_A", "mixed_B", True, None),
            ("mixed_A ↔ mixed_B (mask=1: same)",
             "mixed_A", "mixed_B", None, True),
        ]

        lines = []
        all_passed = True

        for desc, a, b, expect_mask0, expect_mask1 in expectations:
            if expect_mask0 is not None:
                actual = same_cluster(clusters_mask0, a, b)
                passed = actual == expect_mask0
            else:
                actual = same_cluster(clusters_mask1, a, b)
                passed = actual == expect_mask1

            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            line = f"[{status}] {desc}"
            lines.append(line)
            print(line)

        # Write summary
        with open(str(output.summary), "w") as f:
            for line in lines:
                f.write(line + "\n")
            f.write(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")

        if not all_passed:
            raise ValueError("Masking test failed — see details above")


# =============================================================================
# Search masking test
# =============================================================================
# Same expectations as the cluster test, but using ``mmseqs search`` with
# separate query (A sequences) and target (B sequences) databases.
# This verifies that --mask-lower-case works in the search pipeline,
# which we will use for train/val leakage analysis.
# =============================================================================

_SEARCH_QUERY_SUFFIXES = {"_A"}
_SEARCH_TARGET_SUFFIXES = {"_B"}


def _read_fasta(path: str) -> list[tuple[str, str]]:
    """Read a FASTA file into a list of (name, sequence) tuples."""
    records = []
    name = None
    seq_parts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_parts)))
                name = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        records.append((name, "".join(seq_parts)))
    return records


rule split_synthetic_for_search:
    """Split synthetic FASTA into query (A) and target (B) for search test."""
    input:
        fasta="results/tests/synthetic.fasta",
    output:
        query="results/tests/search_query.fasta",
        target="results/tests/search_target.fasta",
    run:
        records = _read_fasta(str(input.fasta))
        query_records = [
            (name, seq) for name, seq in records
            if any(name.endswith(s) for s in _SEARCH_QUERY_SUFFIXES)
        ]
        target_records = [
            (name, seq) for name, seq in records
            if any(name.endswith(s) for s in _SEARCH_TARGET_SUFFIXES)
        ]
        _write_fasta(str(output.query), query_records)
        _write_fasta(str(output.target), target_records)
        print(f"  Query sequences: {[r[0] for r in query_records]}")
        print(f"  Target sequences: {[r[0] for r in target_records]}")


rule run_mmseqs_search_mask_test:
    """Run MMseqs2 search on synthetic data with mask-lower-case={mask_lower_case}.

    Creates separate query/target databases and searches query against target,
    mirroring how the real leakage pipeline will search val against train.
    """
    input:
        query="results/tests/search_query.fasta",
        target="results/tests/search_target.fasta",
    output:
        tsv="results/tests/search_hits_masklc{mask_lower_case}.tsv",
    params:
        query_db="results/tests/search_masklc{mask_lower_case}/queryDB",
        target_db="results/tests/search_masklc{mask_lower_case}/targetDB",
        result_db="results/tests/search_masklc{mask_lower_case}/resultDB",
        tmp="results/tests/search_masklc{mask_lower_case}/tmp",
    wildcard_constraints:
        mask_lower_case="[01]",
    threads: 1
    resources:
        mem_mb=512,
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p $(dirname {params.query_db})
        mmseqs createdb {input.query} {params.query_db} \
            --mask-lower-case {wildcards.mask_lower_case} \
            --threads {threads}
        mmseqs createdb {input.target} {params.target_db} \
            --mask-lower-case {wildcards.mask_lower_case} \
            --threads {threads}
        mmseqs search \
            {params.query_db} \
            {params.target_db} \
            {params.result_db} \
            {params.tmp} \
            --search-type 3 \
            --mask-lower-case {wildcards.mask_lower_case} \
            --min-seq-id 0.5 \
            -c 0.5 \
            --cov-mode 0 \
            --threads {threads}
        mmseqs convertalis \
            {params.query_db} \
            {params.target_db} \
            {params.result_db} \
            {output.tsv} \
            --format-output "query,target,fident,qcov,tcov"
        rm -rf {params.tmp}
        """


rule verify_search_masking_results:
    """Compare MMseqs2 search results with and without --mask-lower-case."""
    input:
        mask0="results/tests/search_hits_masklc0.tsv",
        mask1="results/tests/search_hits_masklc1.tsv",
    output:
        summary="results/tests/search_masking_test_summary.txt",
    run:
        def parse_search_hits(path: str) -> set[tuple[str, str]]:
            """Return set of (query, target) pairs from search results."""
            pairs = set()
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        pairs.add((parts[0], parts[1]))
            return pairs

        def has_hit(hits: set[tuple[str, str]], query: str, target: str) -> bool:
            return (query, target) in hits

        hits_mask0 = parse_search_hits(str(input.mask0))
        hits_mask1 = parse_search_hits(str(input.mask1))

        # Same expectations as the cluster test:
        # - genuine pair: hit in BOTH modes (real homology)
        # - repeat_only pair: hit WITHOUT mask, NO hit WITH mask
        # - mixed pair: hit in BOTH modes (genuine homology survives masking)
        expectations = [
            # (description, query, target, expected_mask0, expected_mask1)
            ("genuine_A → genuine_B (mask=0: hit)",
             "genuine_A", "genuine_B", True, None),
            ("genuine_A → genuine_B (mask=1: hit)",
             "genuine_A", "genuine_B", None, True),
            ("repeat_only_A → repeat_only_B (mask=0: hit)",
             "repeat_only_A", "repeat_only_B", True, None),
            ("repeat_only_A → repeat_only_B (mask=1: no hit)",
             "repeat_only_A", "repeat_only_B", None, False),
            ("mixed_A → mixed_B (mask=0: hit)",
             "mixed_A", "mixed_B", True, None),
            ("mixed_A → mixed_B (mask=1: hit)",
             "mixed_A", "mixed_B", None, True),
        ]

        lines = []
        all_passed = True

        for desc, query, target, expect_mask0, expect_mask1 in expectations:
            if expect_mask0 is not None:
                actual = has_hit(hits_mask0, query, target)
                passed = actual == expect_mask0
            else:
                actual = has_hit(hits_mask1, query, target)
                passed = actual == expect_mask1

            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            line = f"[{status}] {desc}"
            lines.append(line)
            print(line)

        with open(str(output.summary), "w") as f:
            for line in lines:
                f.write(line + "\n")
            f.write(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")

        if not all_passed:
            raise ValueError("Search masking test failed — see details above")


rule test_masking:
    """Target rule: run the full synthetic masking test suite (cluster + search)."""
    input:
        "results/tests/masking_test_summary.txt",
        "results/tests/search_masking_test_summary.txt",
