python shipsnet_newslate_guide_project.py --prior "Gaussian_prior" --epoch 100 --smartpool --no-wd --save-dir "results_shipsnet_01"
python shipsnet_newslate_guide_project.py --prior "Laplace_prior" --epoch 100 --smartpool --no-wd --save-dir "results_shipsnet_01"
python shipsnet_newslate_guide_project.py --prior "Uniform_prior" --epoch 100 --smartpool --no-wd --save-dir "results_shipsnet_01"

python shipsnet_newslate_guide_project.py --prior "Gaussian_prior" --epoch 100 --dropout-mode --no-wd --save-dir "results_shipsnet_02"
python shipsnet_newslate_guide_project.py --prior "Laplace_prior" --epoch 100 --dropout-mode --no-wd --save-dir "results_shipsnet_02"
python shipsnet_newslate_guide_project.py --prior "Uniform_prior" --epoch 100 --dropout-mode --no-wd --save-dir "results_shipsnet_02"

python shipsnet_newslate_guide_project.py --prior "Gaussian_prior" --epoch 100 --wd --save-dir "results_shipsnet_03"
python shipsnet_newslate_guide_project.py --prior "Laplace_prior" --epoch 100 --wd --save-dir "results_shipsnet_03"
python shipsnet_newslate_guide_project.py --prior "Uniform_prior" --epoch 100 --wd --save-dir "results_shipsnet_03"

python shipsnet-experiment05-guide.py --prior "Gaussian_prior" --search-dir "results_GP_shipsnet_newslate_guide" --save-dir "results_GP_shipsnet_newslate_guide_SEU" --limited-mode


python shipsnet-experiment05-guide.py --prior "Gaussian_prior" --search-dir "results_shipsnet_01" --save-dir "results_shipsnet_01_SEU" --limited-mode
python shipsnet-experiment05-guide.py --prior "Laplace_prior" --search-dir "results_shipsnet_01" --save-dir "results_shipsnet_01_SEU" --limited-mode
python shipsnet-experiment05-guide.py --prior "Uniform_prior" --search-dir "results_shipsnet_01" --save-dir "results_shipsnet_01_SEU" --limited-mode


python shipsnet-experiment05-guide.py --prior "Gaussian_prior" --search-dir "results_shipsnet_02" --save-dir "results_shipsnet_02_SEU" --limited-mode
python shipsnet-experiment05-guide.py --prior "Laplace_prior" --search-dir "results_shipsnet_02" --save-dir "results_shipsnet_02_SEU" --limited-mode
python shipsnet-experiment05-guide.py --prior "Uniform_prior" --search-dir "results_shipsnet_02" --save-dir "results_shipsnet_02_SEU" --limited-mode


python shipsnet-experiment05-guide-remedy0.py --prior "Gaussian_prior" --search-dir "results_shipsnet_02" --save-dir "results_shipsnet_02_SEU" --limited-mode --custom-timestamp '_20250720_145610'
python shipsnet-experiment05-guide-remedy0.py --prior "Gaussian_prior" --search-dir "results_shipsnet_02" --save-dir "results_shipsnet_02_SEU" --limited-mode --custom-timestamp '_20250720_105233'


python shipsnet-experiment05-guide.py --prior "Uniform_prior" --search-dir "results_GP_shipsnet_newslate_guide" --save-dir "results_GP_shipsnet_newslate_guide_SEU_check" --limited-mode

python shipsnet-experiment05-guide.py --prior "Uniform_prior" --search-dir "results_shipsnet_01" --save-dir "results_shipsnet_01_SEU_check" --limited-mode
python shipsnet-experiment05-guide.py --prior "Uniform_prior" --search-dir "results_shipsnet_02" --save-dir "results_shipsnet_02_SEU_check" --limited-mode
python shipsnet-experiment05-guide.py --prior "Uniform_prior" --search-dir "results_shipsnet_03" --save-dir "results_shipsnet_03_SEU_check" --limited-mode


python shipsnet_newslate_guide_project_mvrt.py --prior "Gaussian_prior" --epoch 100 --no-wd --save-dir "results_shipsnet_00_mvrt" --var-family "full_rank_multivariate" --b-set 'full'

python shipsnet-experiment05-guide.py --prior "Gaussian_prior" --search-dir "results_shipsnet_00_mvrt" --save-dir "results_shipsnet_00_mvrt_SEU" --limited-mode --multivariate-guide


python shipsnet_newslate_guide_project.py --prior "Uniform_prior" --epoch 100 --bset "single" --save-dir "results_shipsnet_uniform_01"