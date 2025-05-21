import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    """
    Graph Convolution을 위한 기본 노드 합성곱 연산
    Einstein summation을 사용하여 효율적인 행렬 연산 수행
    """
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self, x, A):
        """
        Args:
            x: 입력 노드 특징 [batch, channels, nodes, time]
            A: 인접행렬 [nodes, nodes]
        Returns:
            graph convolution 결과 [batch, channels, nodes, time]
        """
        # Einstein summation: 'ncvl,vw->ncwl'
        # n=batch, c=channels, v=nodes_in, l=time, w=nodes_out
        # x[n,c,v,l] * A[v,w] -> output[n,c,w,l]
        # 차원 변화: [B, C, N, T] * [N, N] -> [B, C, N, T] (채널, 시간 차원 유지)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    """
    1x1 Conv2d를 사용한 선형 변환
    채널 차원을 변경하기 위해 사용
    """
    def __init__(self, c_in, c_out):
        super(linear,self).__init__()
        # 1x1 convolution으로 채널 수 변경
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), 
                                  padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        # 차원 변화: [B, c_in, N, T] -> [B, c_out, N, T] (채널 차원만 변경)
        return self.mlp(x)


class gcn(nn.Module):
    """
    Graph Convolution Network 레이어
    다중 support matrices와 diffusion orders를 지원
    """
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        """
        Args:
            c_in: 입력 채널 수
            c_out: 출력 채널 수
            dropout: 드롭아웃 비율
            support_len: support matrices 개수 (Pf, Pb, Aadp 등)
            order: diffusion order (K, 몇 hop까지 고려할지)
        """
        super(gcn,self).__init__()
        self.nconv = nconv()
        
        # 입력 채널 수 계산
        # order*support_len: 각 support matrix의 k제곱들
        # +1: 원본 x (P^0에 해당)
        c_in = (order*support_len+1)*c_in
        
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        """
        Args:
            x: 입력 특징 [batch, channels, nodes, time]
            support: support matrices 리스트 [Pf, Pb, Aadp, ...]
        Returns:
            GCN 출력 [batch, c_out, nodes, time]
        """
        out = [x]  # P^0*X (원본)
        
        # 각 support matrix에 대해
        for a in support:
            # P^1*X
            # 차원 유지: [B, C, N, T] -> [B, C, N, T]
            x1 = self.nconv(x, a)
            out.append(x1)
            
            # P^2*X, P^3*X, ..., P^order*X
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)  # P^k*X
                out.append(x2)
                x1 = x2
        
        # 모든 diffusion 결과를 채널 차원으로 연결
        # 차원 변화: order*support_len+1개의 [B, C, N, T] -> [B, (order*support_len+1)*C, N, T]
        h = torch.cat(out, dim=1)
        
        # 선형 변환으로 출력 채널 수 맞춤
        # 차원 변화: [B, (order*support_len+1)*C, N, T] -> [B, c_out, N, T]
        h = self.mlp(h)
        
        # 드롭아웃 적용
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    """
    Graph WaveNet 메인 모델
    Dilated Causal Convolution + Graph Convolution을 결합
    """
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, 
                 gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32, skip_channels=256, 
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        """
        Args:
            device: 계산 장치
            num_nodes: 노드 개수 (N) - 그래프의 노드 수 (예: 207개 센서)
            dropout: 드롭아웃 확률
            supports: 사전 정의된 support matrices (Pf, Pb 등)
            gcn_bool: GCN 사용 여부
            addaptadj: 자가적응 인접행렬 사용 여부
            aptinit: 적응적 초기화를 위한 초기 행렬
            in_dim: 입력 특징 차원 (D) - 각 노드의 특성 수 (예: 2개 특성)
            out_dim: 출력 차원 (예측할 시점 수) (T) - 미래 예측 타임스텝 (예: 12 타임스텝)
            residual_channels: residual connection의 채널 수
            dilation_channels: dilated convolution의 채널 수
            skip_channels: skip connection의 채널 수
            end_channels: 마지막 레이어의 채널 수
            kernel_size: convolution kernel 크기
            blocks: block 개수
            layers: 각 block의 layer 개수
        """
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        # 각 레이어의 convolution 모듈들을 저장할 리스트
        self.filter_convs = nn.ModuleList()  # filter branch용
        self.gate_convs = nn.ModuleList()    # gate branch용
        self.residual_convs = nn.ModuleList() # residual connection용
        self.skip_convs = nn.ModuleList()     # skip connection용
        self.bn = nn.ModuleList()            # batch normalization
        self.gconv = nn.ModuleList()         # graph convolution

        # 입력을 residual_channels로 변환하는 첫 번째 convolution
        # 차원 변화: [B, in_dim=2, N, S=12] -> [B, residual_channels=32, N, S=12]
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports
        receptive_field = 1

        # Support matrices 개수 계산
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        # Self Adpative Adj Matrix
        if gcn_bool and addaptadj:
            if aptinit is None:
                # 사전 정의된 초기값이 없는 경우 랜덤 초기화
                if supports is None:
                    self.supports = []
                # E1, E2 node embeddings (수식 5의 E1, E2)
                # 차원: [N, 10] 및 [10, N] - N개 노드를 10차원 임베딩으로 표현
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), 
                                           requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), 
                                           requires_grad=True).to(device)
                self.supports_len += 1
            else:
                # SVD를 사용한 초기화
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                # SVD 결과를 사용해 임베딩 초기화
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        # WaveNet 블록들 구성
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            
            for i in range(layers):
                # Dilated convolution layers
                # Filter branch (tanh 활성화용)
                # 차원 변화: [B, residual_channels=32, N, T] -> [B, dilation_channels=32, N, T-k*d]
                # k: kernel size, d: dilation - 시간 차원이 축소됨
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),
                                                   dilation=new_dilation))

                # Gate branch (sigmoid 활성화용)
                # 차원 변화: [B, residual_channels=32, N, T] -> [B, dilation_channels=32, N, T-k*d]
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), 
                                                 dilation=new_dilation))

                # Residual connection을 위한 1x1 convolution
                # 차원 변화: [B, dilation_channels=32, N, T-k*d] -> [B, residual_channels=32, N, T-k*d]
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # Skip connection을 위한 1x1 convolution
                # 차원 변화: [B, dilation_channels=32, N, T-k*d] -> [B, skip_channels=256, N, T-k*d]
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                
                # Batch normalization
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                # Dilation factor를 매 레이어마다 2배씩 증가
                # 마지막 레이어들에서는 시간 차원이 크게 축소됨 (T=12 -> T=3 등)
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                
                # GCN 레이어 추가
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, 
                                        dropout, support_len=self.supports_len))

        # 최종 출력 레이어들
        # 차원 변화: [B, skip_channels=256, N, T_final=3] -> [B, end_channels=512, N, T_final=3]
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        # 차원 변화: [B, end_channels=512, N, T_final=3] -> [B, out_dim=12, N, T_final=3]
        # out_dim=12는 미래 예측 타임스텝 수
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        """
        Forward pass
        Args:
            input: 입력 데이터 [batch, features=D=2, nodes=N=207, time=S=12]
                  D: 입력 특성 수 (in_dim=2)
                  N: 노드 수 (num_nodes=207)
                  S: 입력 타임스텝 수 (=12, 1시간)
        Returns:
            예측 결과 [batch, out_dim=12, nodes=N=207, T_final=3]
            최종적으로는 [batch, nodes=N=207, out_dim=12]로 변환됨
        """
        in_len = input.size(3)  # 시간 차원의 길이
        
        # Receptive field보다 짧으면 패딩
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
            
        # 시작 convolution으로 채널 수 조정
        # 차원 변화: [B, D=2, N=207, S=12] -> [B, residual_channels=32, N=207, S=12]
        x = self.start_conv(x)
        skip = 0

        # 자가적응 인접행렬 계산 (매 forward pass마다 한 번)
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # 수식 5: Ã_adp = SoftMax(ReLU(E1*E2^T))
            # 차원 변화: [N, 10] * [10, N] -> [N, N] (인접행렬)
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]  # 기존 support에 추가

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # WaveNet 블록 구조:
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            residual = x
            
            # Dilated convolution with gating mechanism
            # 차원 변화: [B, 32, N, T_i] -> [B, 32, N, T_i-k*d]
            # T_i는 각 레이어의 입력 시간 길이, 레이어가 깊어질수록 축소됨
            filter = self.filter_convs[i](residual)  # Filter branch
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)      # Gate branch  
            gate = torch.sigmoid(gate)
            x = filter * gate  # Gated activation

            # Skip connection 계산
            s = x
            # 차원 변화: [B, 32, N, T_i-k*d] -> [B, 256, N, T_i-k*d]
            s = self.skip_convs[i](s)
            try:
                # 시간 차원을 맞춰서 skip connection 누적
                # 레이어마다 시간 차원이 다를 수 있으므로 맞춤
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # Graph convolution 적용
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # 자가적응 인접행렬 포함해서 GCN
                    # 차원 유지: [B, 32, N, T_i-k*d] -> [B, 32, N, T_i-k*d]
                    x = self.gconv[i](x, new_supports)
                else:
                    # 기존 support matrices만 사용
                    x = self.gconv[i](x, self.supports)
            else:
                # GCN 없이 residual convolution만
                # 차원 변화: [B, 32, N, T_i-k*d] -> [B, 32, N, T_i-k*d]
                x = self.residual_convs[i](x)

            # Residual connection
            # 시간 차원을 맞추기 위해 residual의 마지막 부분만 사용
            # 마지막 레이어에서는 T_final=3 정도로 축소됨
            x = x + residual[:, :, :, -x.size(3):]

            # Batch normalization
            x = self.bn[i](x)

        # 최종 출력 계산
        # 차원: [B, 256, N=207, T_final=3]
        x = F.relu(skip)              # Skip connections의 합
        
        # 차원 변화: [B, 256, N=207, T_final=3] -> [B, 512, N=207, T_final=3]
        x = F.relu(self.end_conv_1(x)) # 첫 번째 출력 레이어
        
        # 차원 변화: [B, 512, N=207, T_final=3] -> [B, out_dim=12, N=207, T_final=3]
        x = self.end_conv_2(x)         # 최종 출력 레이어
        
        # 실제 반환 전에 차원 재배열로 [B, N=207, out_dim=12] 형태로 변환하는 경우가 많음
        # (이 코드에는 없지만 사용 단계에서 수행됨)
        return x
