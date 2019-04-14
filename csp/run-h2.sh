#numacmd='numactl --membind 1'
numacmd=''

rm -f log
$numacmd ./main-csp.exe -F data/h2/jac.out -N   500 >> log
$numacmd ./main-csp.exe -F data/h2/jac.out -N  1000 >> log
$numacmd ./main-csp.exe -F data/h2/jac.out -N  2000 >> log
$numacmd ./main-csp.exe -F data/h2/jac.out -N  4000 >> log
$numacmd ./main-csp.exe -F data/h2/jac.out -N  8000 >> log
$numacmd ./main-csp.exe -F data/h2/jac.out -N 10000 >> log
$numacmd ./main-csp.exe -F data/h2/jac.out -N 20000 >> log
$numacmd ./main-csp.exe -F data/h2/jac.out -N 40000 >> log
