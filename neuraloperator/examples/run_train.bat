python train_FNO_diffusion.py const=0.01 use_BC=True
python train_FNO_diffusion.py const=0.02 use_BC=True
python train_FNO_diffusion.py const=0.03 use_BC=True
python train_FNO_diffusion.py const=0.04 use_BC=True
python train_FNO_diffusion.py const=0.01 use_BC=False
python train_FNO_diffusion.py const=0.02 use_BC=False
python train_FNO_diffusion.py const=0.03 use_BC=False
python train_FNO_diffusion.py const=0.04 use_BC=False
python train_FNO_diffusion_mixed.py use_BC=True
python train_FNO_diffusion_mixed.py use_BC=False
python train_FNO_diffusion_mixed.py use_BC=True using_long_data=False
python train_FNO_diffusion_mixed.py use_BC=False using_long_data=False
python train_FNO_diffusion_mixed.py use_BC=True using_long_data=False using_sam=False
python train_FNO_diffusion_mixed.py use_BC=False using_long_data=False using_sam=False

