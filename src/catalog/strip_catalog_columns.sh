# Adapted from https://git.ligo.org/lscsoft/gwcosmo/-/blob/master/scripts_galaxy_catalogs/GLADE%2B/select_columnts.sh
# Columns extracted from the column description of GLADE.
# Selected columns:
# C1 GWGC flag, C2 Hyperleda flag, C3 2Mass, C4Wise, C5 SDSS, C6 Quasar and cluster flag,
# C7 RA, C8 Dec, C9 m_K, C10 err_m_K,
# C11 redshift helio, C12 redshift CMB, C13 Correction flag,
# C14 error peculiar velocity, C15 helio error, C16 redshift and lum distance flag 
awk '{print $3,$4,$5,$6,$7,$8,$9,$10,$19,$20,$28,$29,$30,$31,$32,$35}' GLADE+.txt > GLADE+reduced.txt
