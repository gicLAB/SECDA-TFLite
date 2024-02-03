
import math
def find_len_of_needed_outputs_of_outrows(id,oh,ow,pl,pr,pt,ks,sx,sy):
    width_col = (ow + pl + pr - ks) // sy + 1
    cal_col = id % width_col
    cal_row = id // width_col
    h_pad = -pt + (sy * cal_row)
    w_pad = -pl + (sx * cal_col)
    h_high = h_pad + ks
    w_high = w_pad + ks
    
    h_len = min(h_high, oh) - max(h_pad, 0)
    w_len = min(w_high, ow) - max(w_pad, 0)
    print("h_pad:", h_pad, "h_high:", h_high)
    print("w_pad:", w_pad, "w_high:", w_high)
    print("h_len:", h_len)
    print("w_len:", w_len)
    return h_len * w_len

def col2im(
    depth,
    height,
    width,
    filter_h,
    filter_w,
    pad_t,
    pad_l,
    pad_b,
    pad_r,
    stride_h,
    stride_w,
):
    height_col = (height + pad_t + pad_b - filter_h) // stride_h + 1
    width_col = (width + pad_l + pad_r - filter_w) // stride_w + 1
    h_pad = -pad_t
    im_dex = 0
    map_dex = 0

    wasted_out = 0
    out_map = []
    for h in range(height_col):
        w_pad = -pad_l
        for w in range(width_col):
            im_dex = (h_pad * width + w_pad) * depth
            for ih in range(filter_h):
                for iw in range(filter_w):
                    if (
                        ih + h_pad >= 0
                        and ih + h_pad < height
                        and iw + w_pad >= 0
                        and iw + w_pad < width
                    ):
                        for i in range(depth):
                            map_dex += 1
                            # print(f"{im_dex:4}", ",", end="")
                            out_map.append(im_dex)
                            # if map_dex % ow == 0:
                            #     print("")
                            im_dex += 1
                    else:
                        for i in range(depth):
                            map_dex += 1
                            wasted_out+=1
                            # print(f"{-1:4}", ",", end="")
                            out_map.append(-1)
                            # if map_dex % ow == 0:
                            #     print("")
                            im_dex += 1
                im_dex += depth * (width - filter_w)
            w_pad += stride_w
        h_pad += stride_h

    return out_map, wasted_out



def col2imv2(
    depth,
    height,
    width,
    filter_h,
    filter_w,
    pad_t,
    pad_l,
    pad_b,
    pad_r,
    stride_h,
    stride_w,
):
    height_col = (height + pad_t + pad_b - filter_h) // stride_h + 1
    width_col = (width + pad_l + pad_r - filter_w) // stride_w + 1
    h_pad = -pad_t
    im_dex = 0
    map_dex = 0

    wasted_out = 0
    out_map = []
    for h in range(height_col):
        w_pad = -pad_l
        for w in range(width_col):
            im_dex = (h_pad * width + w_pad) * depth
            for i in range(depth):
                im_dex = (h_pad * width + w_pad) * depth + i
                for ih in range(filter_h):
                    for iw in range(filter_w):
                        if (
                            ih + h_pad >= 0
                            and ih + h_pad < height
                            and iw + w_pad >= 0
                            and iw + w_pad < width
                        ):
                            map_dex += 1
                            out_map.append(im_dex)
                            im_dex += depth
                        else:
                            map_dex += 1
                            wasted_out+=1
                            out_map.append(-1)
                            im_dex += depth
                    im_dex += depth * (width - filter_w)   
            w_pad += stride_w
        h_pad += stride_h

    return out_map, wasted_out

def ComputeOutSize(padding, image_size, filter_size, stride, dilation_rate=1):
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    if stride == 0:
        return 0

    if padding == "same":
        return (image_size + stride - 1) // stride
    elif padding == "valid":
        return (image_size + stride - effective_filter_size) // stride
    else:
        return 0


def compute_padding_with_offset(
    stride, dilation_rate, in_size, filter_size, out_size, offset=0
):
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    total_padding = ((out_size - 1) * stride) + effective_filter_size - in_size
    total_padding = total_padding if total_padding > 0 else 0
    offset = total_padding % 2
    return offset, total_padding // 2
    


def compute_padding_height_width(
    padding,
    stride_height,
    stride_width,
    in_height,
    in_width,
    filter_height,
    filter_width,
):
    dilation_rate_height = 1
    dilation_rate_width = 1

    out_width = ComputeOutSize(
        padding, in_width, filter_width, stride_width, dilation_rate_width,
    )
    out_height = ComputeOutSize(
        padding, in_height, filter_height, stride_height, dilation_rate_height
    )

    offset, p_height = compute_padding_with_offset(
        stride_height, dilation_rate_height, in_height, filter_height, out_height,0
    )
    h_offset = offset
    offset, p_width = compute_padding_with_offset(
        stride_width, dilation_rate_width, in_width, filter_width, out_width, offset
    )
    w_offset = offset
    return p_height, p_width, h_offset, w_offset


def calParams(params):
    stride_x = params[0]
    stride_y = params[1]
    filters = params[2]
    kernel_size = params[3]
    in1 = params[4]
    in2 = params[5]
    in3 = params[6]
    padding_val = params[7]
    out1 = in1 + kernel_size - stride_x
    out2 = in2 + kernel_size - stride_y
    out3 = filters
    rows = filters * kernel_size * kernel_size
    cols = in1 * in2
    depth = in3

    if padding_val == "same":
        out1 = in1 * stride_x
        out2 = in2 * stride_y
    else:
        out1 = in1 + kernel_size - stride_x
        out2 = in2 + kernel_size - stride_y

    ph, pw, pho, pwo = compute_padding_height_width(
        padding_val,
        stride_x,
        stride_y,
        out1,
        out2,
        kernel_size,
        kernel_size,
    )
    pt = ph
    pb = ph + pho
    pl = pw
    pr = pw + pwo
    return rows, cols, depth, out1, out2, out3, pt, pb, pl, pr

def nofSteps(length,stride,kernel_size):
    return int((length- (kernel_size - stride) )/stride) 

# def create_im2col_map(params):
#     stride_x = params[0]
#     stride_y = params[1]
#     filters = params[2]
#     kernel_size = params[3]
#     in1 = params[4]
#     in2 = params[5]
#     in3 = params[6]
#     padding_val = params[7]
#     rows, cols, depth, out1, out2, out3, pt, pb, pl, pr = calParams(params)

#     opos = 0
#     olist = []
#     po_l = out1 + pl + pr 
#     print(po_l)
#     # for  i in range(cols):
#     #     find(opos)



def tconv_model_info(params):
    rows, cols, depth, out1, out2, out3, pt, pb, pl, pr = calParams(params)
    rdepth = math.ceil(depth / 16) * 16
    total_macs = rows * cols * rdepth
    mm2im_out = out1 * out2 * out3
    return (total_macs, mm2im_out)

def custom_tconv_cols(df):
    # add a new column for stride x
    df["stride_x"] = df["model"].str.split("_").str[1].astype(int)
    df["stride_y"] = df["model"].str.split("_").str[2].astype(int)
    df["filters"] = df["model"].str.split("_").str[3].astype(int)
    df["ks"] = df["model"].str.split("_").str[4].astype(int)
    df["ih"] = df["model"].str.split("_").str[5].astype(int)
    df["iw"] = df["model"].str.split("_").str[6].astype(int)
    df["ic"] = df["model"].str.split("_").str[7].astype(int)
    pf = lambda row: (tconv_model_info(
        [
            row["stride_x"],
            row["stride_y"],
            row["filters"],
            row["ks"],
            row["ih"],
            row["iw"],
            row["ic"],
            "same",
        ]
    ))
    df['MACs']= df.apply(pf, axis=1, result_type='expand')[0]
    df['Outputs']= df.apply(pf, axis=1, result_type='expand')[1]
    df["Compute Intensity"] = df["MACs"] / df["Outputs"]
    return df