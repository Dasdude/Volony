def linear_interp(x_min,y_min,x_max,y_max,x_max_bound,y_max_bound):
    slope = float(y_max-y_min)/float(x_max-x_min)
    bias = y_max -slope*x_max
    x_max_clip = min(x_max,x_max_bound)
    y_max_clip = min(y_max,y_max_bound)
    x_min_clip = max(x_min,0)
    y_min_clip = max(y_min,0)
    y_xmin = (slope*x_min_clip)+bias
    y_xmax = (slope*x_max_clip)+bias
    x_ymin = (y_min_clip-bias)/slope
    x_ymax=  (y_max_clip-bias)/slope
    if is_inbound(y_xmin,0,y_max_bound):
        res_y_min = y_xmin
        res_x_min = x_min_clip
    if is_inbound(x_ymin, 0, x_max_bound):
        res_y_min = y_min_clip
        res_x_min = x_ymin
    if is_inbound(y_xmax,0,y_max_bound):
        res_y_max = y_xmax
        res_x_max = x_max_clip
    if is_inbound(x_ymax, 0, x_max_bound):
        res_y_max = y_max_clip
        res_x_max = x_ymax
    return res_x_min,res_y_min,res_x_max,res_y_max
def is_inbound(val,min_bound,max_bound):
    if val>=min_bound and val<=max_bound:
        return True
    else:
        return False
if __name__ == '__main__':
    linear_interp(2,3,8,15,3,3,7,15)