import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
"""
Original code from https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch/blob/master/CompactBilinearPooling.py
Hacked by Sungyong Baik (dsybaik@snu.ac.kr)
"""


class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.

    Args:

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.

        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.

        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.

        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, cuda=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = nn.Parameter(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim), requires_grad=False)

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = nn.Parameter(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim), requires_grad=False)

        if cuda:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2

    def forward(self, bottom1, bottom2):
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        fft1 = torch.fft(torch.cat((sketch_1.unsqueeze(-1), torch.zeros(sketch_1.size()).unsqueeze(-1).cuda()), -1), 1)
        fft2 = torch.fft(torch.cat((sketch_2.unsqueeze(-1), torch.zeros(sketch_2.size()).unsqueeze(-1).cuda()), -1), 1)

        fft1_real = fft1[..., 0]
        fft1_imag = fft1[..., 1]
        fft2_real = fft2[..., 0]
        fft2_imag = fft2[..., 1]

        temp_rr, temp_ii = fft1_real.mul(fft2_real), fft1_imag.mul(fft2_imag)
        temp_ri, temp_ir = fft1_real.mul(fft2_imag), fft1_imag.mul(fft2_real)
        fft_product_real = temp_rr - temp_ii
        fft_product_imag = temp_ri + temp_ir

        cbp_flat = torch.ifft(torch.cat((fft_product_real.unsqueeze(-1), fft_product_imag.unsqueeze(-1)), -1), 1)
        cbp_flat = cbp_flat[..., 0]

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)*self.output_dim

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)
        else:
            cbp = cbp.permute(0,3,1,2)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()

class Bilinear_Pooling(nn.Module):
    def __init__(self, num_feat1=512, num_feat2=128, num_feat_out=512):
        super(Bilinear_Pooling, self).__init__()
        self.num_feat1 = num_feat1
        self.num_feat2 = num_feat2
        self.num_feat_out = num_feat_out

        self.cbp = CompactBilinearPooling(self.num_feat1, self.num_feat2, self.num_feat_out, sum_pool=True)

    def forward(self, input1, input2):
        assert(self.num_feat1 == input1.shape[1])
        assert(self.num_feat2 == input2.shape[1])
        output = self.cbp(input1, input2)
        output = F.normalize(output, dim=1)
        output = output.squeeze()
        return output
if __name__ == '__main__':

    bottom1 = Variable(torch.randn(2, 512, 14, 14))
    bottom2 = Variable(torch.randn(2, 512, 14, 14))

    layer = CompactBilinearPooling(512, 512, 512)
#    layer
#    layer.train()

    out = layer(bottom1, bottom2)
    print(out.shape)

