# HOXD55 substitution scoring matrix — more permissive divergent-species
# variant of HOXD70. Used by UCSC for the most divergent pairwise mammal
# alignments. Slightly-weaker mismatch penalties encourage hitting more
# lower-identity seeds.
#
# Reference: Chiaromonte, Yap & Miller (2002), Pac. Symp. Biocomput. 7:115-126.
# Gap penalties are supplied via --gap= on the CLI.

     A    C    G    T
A    91  -90  -25 -100
C   -90  100 -100  -25
G   -25 -100  100  -90
T  -100  -25  -90   91
