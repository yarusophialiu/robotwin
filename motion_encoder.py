class MotionEncoder3D(nn.Module):
    def __init__(self, in_ch=2, out_dim=96): # ↓ from 256 TODO 96 128
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv3d(in_ch, 16, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)), nn.GELU(),
        nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(2,2,2), padding=1), nn.GELU(),
        nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=1), nn.GELU(),
        nn.AdaptiveAvgPool3d((None,1,1))
        )
        self.proj = nn.Linear(64, out_dim) # ↓ from 128→256

    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        x = self.net(x) # [B,64,T',1,1]
        x = x.squeeze(-1).squeeze(-1) # [B,64,T']
        x = self.proj(x.transpose(1,2)) # [B,T',out_dim]
        return x
    given the architecture, can you write the section of resolution predictor? 



