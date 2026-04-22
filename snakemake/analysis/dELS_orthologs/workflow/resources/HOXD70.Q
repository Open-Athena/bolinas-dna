# HOXD70 substitution scoring matrix — the standard divergent cross-species
# matrix (used by UCSC for hg/mm lastz chain/net). Human-mouse ~25% divergent
# non-coding is well inside its design range.
# Reference: Chiaromonte, Yap & Miller (2002), Pac. Symp. Biocomput. 7:115-126.
#
# Gap penalties are supplied on the lastz CLI via --gap=400,30 rather than here.

     A    C    G    T
A    91 -114  -31 -123
C  -114  100 -125  -31
G   -31 -125  100 -114
T  -123  -31 -114   91
