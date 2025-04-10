import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    # Default values for binarization of grayscale images
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder path
        for feature in features:
            self.encoder.append(self.conv_block(in_channels, feature))
            in_channels = feature
        
        # U-Net bottom layer
        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)
        
        # Decoder path
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.conv_block(feature * 2, feature))
        
        # Output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        skip_connections = []
        
        # Econder path
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse the skip connections
        
        # Decoder path
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i // 2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](x)
            
        return torch.sigmoid(self.final_conv(x))
    
# Testing the model
if __name__ == "__main__":
    model = UNet()
    x = torch.randn((1, 1, 256, 256))  # Example input (Batch size, Channels, Height, Width)
    preds = model(x)
    print(preds.shape)  # Should be (1, 1, 256, 256)
        