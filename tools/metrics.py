import numpy as np
import torch
class Eval_tool():
    def __init__(self):
        self.mae = 0.0
        self.avgF=0.0
        self.maxF = 0.0
        self.meanF= 0.0
        self.Smeasure = 0.0
        self.count = 0.0
        self.beta2 = 0.3
        self.alpha= 0.5
    def run_eval(self, pred, gt):
        self.mae += self.cal_mae(pred, gt)
        self.avgF += self.cal_fmeasure(pred, gt)
        self.Smeasure += self.cal_Smeasure(pred, gt)
        self.count +=1
    def get_score(self):
        self.mae /= self.count
        self.maxF = (self.avgF/self.count).max().item()
        self.Smeasure /= self.count
        return self.mae, self.maxF,self.Smeasure
    def reset(self):
        self.mae = 0.0
        self.avgF=0.0
        self.maxF = 0.0
        self.meanF= 0.0
        self.Smeasure = 0.0
        self.count = 0.0
        self.beta2 = 0.3
        self.alpha= 0.5 
    def cal_mae(self, pred, gt):
        mea = torch.abs(pred - gt).mean()
        return mea.item()
    def cal_fmeasure(self, pred, gt):
        prec, recall = self._eval_pr(pred, gt, 255)
        f_score = (1 + self.beta2) * prec * recall / (self.beta2 * prec + recall)
        f_score[f_score != f_score] = 0
        return f_score
    def cal_Smeasure(self, pred, gt):
        y = gt.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            Q = self.alpha * self._S_object(pred, gt) + (1-self.alpha) * self._S_region(pred, gt)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        return Q.item()


    def _eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall
    
    def _S_object(self, pred, gt):
        fg = torch.where(gt==0, torch.zeros_like(pred), pred)
        bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        
        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        # print(Q)
        return Q
    
    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            i = torch.from_numpy(np.arange(0,cols)).cuda().float()
            j = torch.from_numpy(np.arange(0,rows)).cuda().float()
            X = torch.round((gt.sum(dim=0)*i).sum() / total)
            Y = torch.round((gt.sum(dim=1)*j).sum() / total)
        return X.long(), Y.long()
    
    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h*w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
        
        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q
