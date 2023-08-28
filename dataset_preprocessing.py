import argparse
import os
import json
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi




def Frame_Length(x, overlap, nwind):
    nx = len(x)
    noverlap = nwind - overlap
    framelen = int((nx - noverlap) / (nwind - noverlap))
    return framelen


def Truelabel2Trueframe(TrueLabel_bin, wsize, wstep):
    iidx = 0
    Frame_iidx = 0
    Frame_len = Frame_Length(TrueLabel_bin, wsize, wstep)
    Detect = np.zeros([Frame_len, 1])
    while 1:
        if iidx + wstep <= len(TrueLabel_bin):
            TrueLabel_frame = TrueLabel_bin[iidx : iidx + wstep - 1] * 10
        else:
            TrueLabel_frame = TrueLabel_bin[iidx:] * 10

        if np.sum(TrueLabel_frame) >= wstep / 2:
            TrueLabel_frame = 1
        else:
            TrueLabel_frame = 0

        if Frame_iidx >= len(Detect):
            break

        Detect[Frame_iidx] = TrueLabel_frame
        iidx = iidx + wsize
        Frame_iidx = Frame_iidx + 1
        if iidx > len(TrueLabel_bin):
            break
    return Detect


def sil2lab(wavlab):
    sil = {}
    with open(wavlab, "r", encoding="utf-8") as inf:
        for file in inf.readlines():
            utt, file1 = file.strip().split()
            file_ = open(file1, "r")
            d = set()
            sil[utt] = d
            for line in file_.readlines():
                st, end, lab = line.strip().split()
                if lab == "h#":
                    sil[utt].add(int(st))
                    sil[utt].add(int(end))
            sil[utt] = sorted(sil[utt])
    return sil


def integrate_utt_lab(wav_scp, label_path):
    data = {}
    file_name = wav_scp.split("/")[-2]
    out_dr = file_name+"_data.list"
    print(out_dr)
    os.system(f"rm -rf ./{out_dr}")
    datalist = open(f"./{out_dr}", "a")
    with open(wav_scp, "r", encoding="utf-8") as inf:
        utt_lab = sil2lab(label_path)
        utt_ = [(line.strip().split()) for line in inf.readlines()]

        for i, k in zip(utt_, utt_lab):
            if i[0] == k:
                wavform, sr = torchaudio.load(i[1])
                data["key"] = i[0]
                data["wav"] = i[1]
                label = torch.zeros(wavform.size(1))
                (
                    label[: utt_lab[k][1]],
                    label[utt_lab[k][1] : utt_lab[k][2] ],
                    label[utt_lab[k][3] :],
                ) = (0, 1, 0)
                
                mfcc = kaldi.fbank(
                wavform, num_mel_bins=23, frame_length=25, frame_shift=10, dither=0.0
        )
                assert Frame_Length(wavform[0].numpy(), 160, 400)== mfcc.size(0)

                data["frame_label"] = torch.from_numpy(
                                Truelabel2Trueframe(label.numpy(), 160, 400)
                                ).squeeze(1).int().tolist()
              
                json_line = json.dumps(data, ensure_ascii=False)
                datalist.write(json_line + "\n")



def frame2rawlabel(label, win_len, win_step):
    num_frame = label.shape[0]

    total_len = (num_frame - 1) * win_step + win_len
    raw_label = np.zeros((total_len, 1))
    start_indx = 0

    i = 0

    while True:
        if start_indx + win_len > total_len:
            break
        else:
            temp_label = label[i]
            raw_label[start_indx + 1 : start_indx + win_len] = (
                raw_label[start_indx + 1 : start_indx + win_len] + temp_label
            )
        i += 1

        start_indx = start_indx + win_step

    raw_label = (raw_label >= 1).choose(raw_label, 1)

    return raw_label


if __name__ == "__main__":

    import sys
    # [wav_label, wav_scp] = sys.argv[1:]
    
    wav_label = "/home/junlin/wenet/examples/timit/data/dev/wav.label"
    wav_scp = "/home/junlin/wenet/examples/timit/data/dev/wav.scp"
    
    integrate_utt_lab(wav_scp, wav_label)