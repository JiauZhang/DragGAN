import torch, math

def linear(feature, p0, p1, d, axis=0):
    f0 = feature[..., p0[0], p0[1]]
    f1 = feature[..., p1[0], p1[1]]
    weight = abs(d[axis])
    f = (1 - weight) * f0 + weight * f1
    return f

def bilinear(feature, qi, d):
    x0, y0 = qi
    dx, dy = d
    dx = 1 if dx >= 0 else -1
    dy = 1 if dy >= 0 else -1
    x1 = x0 + dx
    y1 = y0 + dy
    fx1 = linear(feature, (x0, y0), (x1, y0), d, axis=0)
    fx2 = linear(feature, (x0, y1), (x1, y1), d, axis=0)
    weight = abs(d[1])
    fx = (1 - weight) * fx1 + weight * fx2
    return fx

def motion_supervision(F0, F, pi, ti, r1=3, M=None):
    loss = 0
    dx, dy = ti[0] - pi[0], ti[1] - pi[1]
    norm = math.sqrt(dx**2 + dy**2)
    d = (dx / norm, dy / norm)

    for x in range(pi[0] - r1, pi[0] + r1):
        for y in range(pi[1] - r1, pi[1] + r1):
            qi = (x, y)
            loss += torch.mean(torch.abs(
                F[..., qi[0], qi[1]].detach() - bilinear(F, qi, d)
            ))

    return loss

def point_tracking(F0, F, pi, r2=12):
    diff = 1e8
    npi = pi
    with torch.no_grad():
        for x in range(pi[0] - r2, pi[0] + r2):
            for y in range(pi[1] - r2, pi[1] + r2):
                diff_ = torch.norm(torch.abs(
                    F0[..., pi[0], pi[1]] - F[..., x, y]
                ))
                if diff > diff_:
                    diff = diff_
                    npi = (x, y)
    return npi
