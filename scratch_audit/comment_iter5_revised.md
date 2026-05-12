🤖 **Iter 5 — Training-weight impact + Zoonomia's actual masking pipeline**

Two pieces of new context from the user:

1. The training pipeline tokenises `a` and `A` identically but **downweights repeat (lowercase) bases by 100×** during loss / gradient accumulation. So per-species masking variance directly biases per-species training contribution.
2. My iter-4 "Cactus's own mask is added on top" line was hand-waving. After tracing through the published build documentation, the actual pipeline is more interesting — and it changes the fix-option ranking.

## What Zoonomia actually does for masking

Sources:

- [`cactus/doc/progressive.md`](https://github.com/ComparativeGenomicsToolkit/cactus/blob/master/doc/progressive.md): `cactus-preprocess` runs **RED** (alignment-based repeat detection) by default. The doc explicitly says it **adds** to existing soft-mask rather than replacing it: "We do some basic masking as a preprocessing step to ensure highly repetitive elements are masked when repeat libraries are incomplete."
- [241-mammalian-2020v2.1.README.md](https://cgl.gi.ucsc.edu/data/cactus/241-mammalian-2020v2.1.README.md): "New genomes from GenBank were **pre-masked with RepeatMasker using the repeatMaskerPipeline tool** … which employs **Dfam version 3.3**" and "Masked assemblies produced about **40-50% of the genome softmasked**, like we expect from UCSC and what was used for the original Zoonomia data."
- [447-mammalian-2022v1.README.md](https://cgl.gi.ucsc.edu/data/cactus/447-mammalian-2022v1.README.md): 447 = 241-mammalian + 243-primate joined via bridge construction. Doesn't re-document preprocessing — inherits from the constituent alignments.

So the pipeline is: **uniform RepeatMasker pre-mask (Dfam 3.3, target ~40-50% lowercase per assembly) → cactus-preprocess RED step (adds alignment-based repeats on top) → HAL**.

This explains iter-4's HAL ≥ NCBI Datasets pattern: HAL has `RepeatMasker(Dfam 3.3, uniform protocol) ∪ RED` while NCBI Datasets serves whatever the original submitter chose to mask. For GCF leaves where NCBI did its own thorough job, the HAL adds RED's contribution on top (e.g., Ceratotherium 53.8% HAL vs 29.6% NCBI). For Homo_sapiens, NCBI's current mask is essentially equivalent to Zoonomia's protocol output (38.0% vs 35.6%) — RED contributed only marginally.

## Training-weight impact under 100× downweighting

Effective weight per base under the policy `weight = (1 − lower − N)·1.0 + lower·0.01 + N·0.0`:

| species | acc | prefix | lower% | N% | **effective weight** |
|---|---|---|---:|---:|---:|
| Lycaon_pictus | GCA_001887905.1 | GCA | 19.1% | 3.03% | **0.7803** |
| Ursus_maritimus | GCF_000687225.1 | GCF | 19.9% | 0.87% | 0.7946 |
| Ceratotherium_simum | GCF_000283155.1 | GCF | 19.1% | 0.27% | 0.8081 |
| … | | | | | … |
| Petromus_typicus | GCA_004026965.1 | GCA | 3.9% | 0.01% | 0.9611 |
| Homo_sapiens | GCA_000001405.27 | GCA | 1.9% | 0.00% | **0.9813** |

Range 0.78–0.98 across 108 species (**1.26× spread**); mean 0.91.

By accession prefix:

| accession prefix | n species | mean effective weight |
|---|---:|---:|
| GCF | 29 | **0.832** |
| GCA | 79 | **0.934** |

**Per-prefix gap: 10 pp.** GCA leaves contribute ~12% more gradient per base than GCF leaves. Three contributors:

1. **Dfam 3.3 library coverage varies by lineage.** Well-characterised mammals (mouse, dog, cattle, primates) have richer repeat families catalogued; obscure rodents, bats, marine mammals have thinner Dfam coverage. RepeatMasker masks less in those leaves regardless of protocol uniformity.
2. **The conservation-filter recipe biases anchors toward repeat-poor regions.** Even at the whole-genome level the HAL Petromus_typicus is 37.4% lowercase — within the 40-50% Zoonomia target — but its v1 anchors drop to 3.9% because conservation-filtered windows preferentially land in CDS/UTR/UCE-like regions, which are repeat-poor on every leaf.
3. **Per-species conservation profile.** Different mammals have different distributions of where conservation peaks fall; some species' conserved regions overlap more repeat-rich neighbourhoods than others.

Components (1) and (3) are the assembly-source-driven variance; (2) is the recipe.

## Re-ranked fix options

R1 is the most defensible answer **now that we know v1's mask is already what a uniform-RepeatMasker pipeline produces given current Dfam coverage.**

| ID | option | comments |
|---|---|---|
| **R1** *(recommended)* | Accept + document in the dataset card. v1's masking = `RepeatMasker(Dfam 3.3, uniform) ∪ RED`. Per-species lowercase variance is mostly recipe-bias + Dfam-coverage-by-lineage; not a pipeline defect. The 10 pp GCF/GCA gap in v1 anchors is real but small relative to the 80–98% of bases that aren't downweighted at all. | nil cost |
| **R5** *(new — modest)* | Re-mask each species with newer **Dfam 3.9** (~80 GB library; Zoonomia used 3.3 from 2018). Dfam 3.9 ships substantially more families for under-represented mammalian lineages, would lift the lower-quartile species from 4% to maybe 7-10% lowercase, reducing the GCF/GCA gap somewhat. Wouldn't move well-masked leaves. | Cost: 108 species × ~30 min/species on 1 CPU = ~54 CPU-hours total; ~3 h wall on a c6id.4xlarge. Library download is the one-off cost. Worth a 1-species spike (e.g., Petromus_typicus or another bottom-quartile leaf) to gauge actual delta before committing. |
| R2 | "Run RepeatMasker uniformly." Already done by the Zoonomia protocol. Re-running with the same Dfam library would be a no-op. | superseded by R5 |
| R3 | Per-species RepeatMasker BEDs from UCSC/Ensembl/NCBI. | Heterogeneous availability; won't beat R5 systematically |
| ~~R4~~ | ~~Re-extract from NCBI Datasets.~~ Rejected in iter-4. | |

**Recommendation: R1 if the 10 pp / 1.26× training-weight asymmetry is acceptable; R5 if you want to chip away at it cheaply.** R2's premise (the dataset hasn't been uniformly masked) is wrong — it has been, just with a 2018-era Dfam library.

## What's still unknown

- **Empirical impact on the model.** A 10 pp per-prefix effective-weight bias may or may not move downstream eval scores. Train two versions and measure is the only way; out of scope here.
- **Dfam 3.9 vs 3.3 delta.** Worth a one-species spike before R5 is a serious option. Pick a low-lowercase GCA leaf (e.g., Petromus_typicus at 3.9%, or one of the GCA_004… Zoonomia-supp species) and run both library versions on the HAL-derived FASTA. If 3.9 doubles lowercase on those leaves, R5 is worth doing. If it moves it by 0.5 pp, it isn't.
