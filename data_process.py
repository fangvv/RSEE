import numpy as np
import pandas as pd

templ = "info_QA"
a = np.load(templ + ".npy").astype(int)

sample_num = a.shape[0]
if sample_num == 1530:
    num_class = 51
else:
    num_class = 101
num_frames = a.shape[1] - 2

resos = np.array([224, 168, 112, 84])

all_class_cnt = np.zeros((num_class))
hit_class_cnt = np.zeros((num_class))

all_class_used_frame = np.zeros((num_class))
all_class_reso_cnt = np.zeros((num_class))
all_class_reso_distribute = np.zeros((num_class, resos.shape[0])) # 224 168, ...
# -1 for GT , -2 for Pred
for i, item in enumerate(a):
    current_sample = a[i]
    all_class_cnt[current_sample[-1]] += 1
    class_idx = current_sample[-1]
    if current_sample[-2] == current_sample[-1]:
        hit_class_cnt[current_sample[-2]] += 1
    neg_one_idx = np.where(current_sample == -1)
    if neg_one_idx[0].size != 0:
        reso_num = np.where(current_sample == -1)[0][0]
    else:
        reso_num = num_frames
    all_class_used_frame[class_idx] += reso_num
    current_sample[current_sample == -1] = 0
    reso_only = current_sample[:reso_num]
    for item in reso_only:
        all_class_reso_distribute[class_idx, item] += 1
    reso_exact = resos[reso_only]
    all_class_reso_cnt[class_idx] += reso_exact.sum() / reso_num

    

prec_per_cls = hit_class_cnt / all_class_cnt
used_frame_per_cls = all_class_used_frame / all_class_cnt
reso_per_cls = all_class_reso_cnt / (all_class_cnt) 


out = np.concatenate((np.expand_dims(prec_per_cls, axis=1), np.expand_dims(used_frame_per_cls, axis=1), np.expand_dims(reso_per_cls, axis=1)), axis=1)
print(out, all_class_reso_distribute)
print(all_class_reso_distribute.transpose().shape)

def save_to_excel(a, reso_distri):
    path = templ + '.xlsx'
    with pd.ExcelWriter(path) as writer:
        df_info = pd.DataFrame(
            ([
                [b[0], b[1], b[2]]
                for b in a
            ])
        )
        df_info.to_excel(writer, sheet_name='info', header=False, index=False)

        # for [101, 4]
        # df_info_1 = pd.DataFrame(
        #     ([
        #         [b[0], b[1], b[2], b[3]]
        #         for b in reso_distri
        #     ])
        # )

        # for [4, 101]
        df_info_1 = pd.DataFrame(
            ([
                [b[i] for i in range(reso_distri.shape[1])]
                for b in reso_distri
            ])
        )
        # print(df_info_1)
        df_info_1.to_excel(writer, sheet_name='reso_distri', header=False, index=False)
save_to_excel(out, all_class_reso_distribute.transpose())