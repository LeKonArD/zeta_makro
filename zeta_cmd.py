#!/usr/bin/python3

import argparse
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing as prp


# functions from pyzeta

def calculate_scores(docprops1, docprops2, relfreqs1, relfreqs2, absolute1, absolute2, logaddition, segment_length):
    """
    This function implements several variants of Zeta by modifying some key parameters.
    Scores can be document proportions (binary features) or relative frequencies.
    Scores can be taken directly or subjected to a log-transformation (log2, log10)
    Scores can be subtracted from each other or divided by one another.
    The combination of document proportion, no transformation and subtraction is Burrows' Zeta.
    The combination of relative frequencies, no transformation, and division corresponds to
    the ratio of relative frequencies.
    """
    # Define logaddition and division-by-zero avoidance addition
    logaddition = logaddition
    divaddition = 0.00000000001
    # == Calculate subtraction variants ==
    # sd0 - Subtraction, docprops, untransformed a.k.a. "original Zeta"
    sd0 = docprops1 - docprops2
    sd0 = pd.Series(sd0, name="sd0")
    # Prepare scaler to rescale variants to range of sd0 (original Zeta)
    scaler = prp.MinMaxScaler(feature_range=(min(sd0), max(sd0)))
    # sd2 - Subtraction, docprops, log2-transformed
    sd2 = np.log2(docprops1 + logaddition) - np.log2(docprops2 + logaddition)
    sd2 = pd.Series(sd2, name="sd2")
    sd2 = scaler.fit_transform(sd2.values.reshape(-1, 1))
    # sdX - Subtraction, docprops, log10-transformed
    sdX = np.log10(docprops1 + logaddition) - np.log10(docprops2 + logaddition)
    sdX = pd.Series(sdX, name="sdX")
    sdX = scaler.fit_transform(sdX.values.reshape(-1, 1))
    # sr0 - Subtraction, relfreqs, untransformed
    sr0 = relfreqs1 - relfreqs2
    sr0 = pd.Series(sr0, name="sr0")
    sr0 = scaler.fit_transform(sr0.values.reshape(-1, 1))
    # sr2 - Subtraction, relfreqs, log2-transformed
    sr2 = np.log2(relfreqs1 + logaddition) - np.log2(relfreqs2 + logaddition)
    sr2 = pd.Series(sr2, name="sr2")
    sr2 = scaler.fit_transform(sr2.values.reshape(-1, 1))
    # srX - Subtraction, relfreqs, log10-transformed
    srX = np.log10(relfreqs1 + logaddition) - np.log10(relfreqs2 + logaddition)
    srX = pd.Series(srX, name="srX")
    srX = scaler.fit_transform(srX.values.reshape(-1, 1))

    # == Division variants ==
    # dd0 - Division, docprops, untransformed
    dd0 = (docprops1 + divaddition) / (docprops2 + divaddition)
    dd0 = pd.Series(dd0, name="dd0")
    dd0 = scaler.fit_transform(dd0.values.reshape(-1, 1))
    # dd2 - Division, docprops, log2-transformed
    dd2 = np.log2(docprops1 + logaddition) / np.log2(docprops2 + logaddition)
    dd2 = pd.Series(dd2, name="dd2")
    dd2 = scaler.fit_transform(dd2.values.reshape(-1, 1))
    # ddX - Division, docprops, log10-transformed
    ddX = np.log10(docprops1 + logaddition) / np.log10(docprops2 + logaddition)
    ddX = pd.Series(ddX, name="ddX")
    ddX = scaler.fit_transform(ddX.values.reshape(-1, 1))
    # dr0 - Division, relfreqs, untransformed
    dr0 = (relfreqs1 + divaddition) / (relfreqs2 + divaddition)
    dr0 = pd.Series(dr0, name="dr0")
    dr0 = scaler.fit_transform(dr0.values.reshape(-1, 1))
    # dr2 - Division, relfreqs, log2-transformed
    dr2 = np.log2(relfreqs1 + logaddition) / np.log2(relfreqs2 + logaddition)
    dr2 = pd.Series(dr2, name="dr2")
    dr2 = scaler.fit_transform(dr2.values.reshape(-1, 1))
    # drX - Division, relfreqs, log10-transformed
    drX = np.log10(relfreqs1 + logaddition) / np.log10(relfreqs2 + logaddition)
    drX = pd.Series(drX, name="drX")
    drX = scaler.fit_transform(drX.values.reshape(-1, 1))

    # Calculate Gries "deviation of proportions" (DP)
    segnum1 = len(absolute1.columns.values)
    segnum2 = len(absolute2.columns.values)
    seglens1 = [segment_length] * segnum1
    seglens2 = [segment_length] * segnum2
    crpsize1 = sum(seglens1)
    crpsize2 = sum(seglens2)

    totalfreqs1 = np.sum(absolute1, axis=1)
    totalfreqs2 = np.sum(absolute2, axis=1)

    expprops1 = np.array(seglens1) / crpsize1
    expprops2 = np.array(seglens2) / crpsize2

    obsprops1 = absolute1.div(totalfreqs1, axis=0)
    obsprops1 = obsprops1.fillna(expprops1[0])  # was: expprops1[0]
    obsprops2 = absolute2.div(totalfreqs2, axis=0)
    obsprops2 = obsprops2.fillna(expprops2[0])  # was: expprops2[0]
    devprops1 = (np.sum(abs(expprops1 - obsprops1), axis=1) / 2)
    devprops2 = (np.sum(abs(expprops2 - obsprops2), axis=1) / 2)

    # Calculate DP variants ("g" for Gries)
    sg0 = devprops1 - devprops2
    sg0 = pd.Series(sg0, name="sg0")
    sg0 = scaler.fit_transform(sg0.values.reshape(-1, 1))
    sg2 = np.log2(devprops1 + logaddition) - np.log2(devprops2 + logaddition)
    sg2 = pd.Series(sg2, name="sg2")
    sg2 = scaler.fit_transform(sg2.values.reshape(-1, 1))
    dg0 = (devprops1 + divaddition) / (devprops2 + divaddition)
    dg0 = pd.Series(dg0, name="dg0")
    dg0 = scaler.fit_transform(dg0.values.reshape(-1, 1))
    dg2 = np.log2(devprops1 + logaddition) / np.log2(devprops2 + logaddition)
    dg2 = pd.Series(dg2, name="dg2")
    dg2 = scaler.fit_transform(dg2.values.reshape(-1, 1))

    # Return all zeta variant scores
    return sd0, sd2.flatten(), sr0.flatten(), sr2.flatten(), sg0.flatten(), sg2.flatten(), dd0.flatten(), dd2.flatten(), dr0.flatten(), dr2.flatten(), dg0.flatten(), dg2.flatten(), devprops1, devprops2


def get_meanrelfreqs(relative):
    meanrelfreqs = np.mean(relative, axis=1) * 1000

    return meanrelfreqs


def combine_results(docprops1, docprops2, relfreqs1, relfreqs2, devprops1, devprops2, meanrelfreqs, sd0, sd2, sr0, sr2,
                    sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2):
    results = pd.DataFrame({
        "docprops1": docprops1,
        "docprops2": docprops2,
        "relfreqs1": relfreqs1,
        "relfreqs2": relfreqs2,
        "devprops1": devprops1,
        "devprops2": devprops2,
        "meanrelfreqs": meanrelfreqs,
        "sd0": sd0,
        "sd2": sd2,
        "sr0": sr0,
        "sr2": sr2,
        "sg0": sg0,
        "sg2": sg2,
        "dd0": dd0,
        "dd2": dd2,
        "dr0": dr0,
        "dr2": dr2,
        "dg0": dg0,
        "dg2": dg2})
    results = results[[
        "docprops1",
        "docprops2",
        "relfreqs1",
        "relfreqs2",
        "devprops1",
        "devprops2",
        "meanrelfreqs",
        "sd0",
        "sd2",
        "sr0",
        "sr2",
        "sg0",
        "sg2",
        "dd0",
        "dd2",
        "dr0",
        "dr2",
        "dg0",
        "dg2"]]
    results.sort_values(by="sg0", ascending=False, inplace=True)
    # print(results.head(10), "\n", results.tail(10))
    return results


def save_results(results, resultsfile):
    with open(resultsfile, "w") as outfile:
        results.to_csv(outfile, sep="\t")


# dnb zeta functions

def read_meta(m_path):

    meta = pd.read_csv(m_path, sep="\t", low_memory=False)
    meta = meta[meta["seq_dtm"] == 1]
    meta = meta[meta["token_border"] == 1]
    meta = meta[meta["quality"] == 1]

    return meta


def remove_stopwords(seg, stopwords_file):

    print("remove stopwords ..")
    stopwords = pd.read_csv(stopwords_file, sep="\t", header=None)
    stopwords = list(stopwords.iloc[:, 0])
    rev_stoplist = [i for i in seg.index if i not in stopwords]
    seg = seg.loc[rev_stoplist]

    return seg


def get_segments_focus_group(genre, meta, max_segments_per_group, stopwords_file, segmentlen):

    segments_group_a = pd.DataFrame()
    meta_group_a = meta[meta["genre"] == genre]
    i = 0
    c = 1
    for index, row in meta_group_a.iterrows():

        if c >= max_segments_per_group:
            break

        dtm_path = path_to_seq_dtm + row["folder"] + "/" + row["id"]
        print(dtm_path)
        data = pd.read_csv(dtm_path, sep="\t", dtype={"token": str, "count": np.int32})

        for seq_id in np.unique(data.iloc[:, 0]):

            seq = data[data.iloc[:, 0] == seq_id][["token", "count"]]
            seq.index = seq["token"]
            seq = seq.rename(columns={"count": genre + "_" + str(i) + "_" + str(seq_id)})
            seq = seq.drop("token", axis=1)
            segments_group_a = pd.concat([segments_group_a, seq], axis=1, sort=False).fillna(0)
            if c >= max_segments_per_group:
                break

            c += 1
        i += 1

    segments_group_a = remove_stopwords(segments_group_a, stopwords_file)

    return segments_group_a


def get_segments_comp_group(comparatives_group, segments_per_group, stopwords, selection_mode):

    segments_group = pd.DataFrame()
    meta_group_b = meta[meta["genre"] != comparatives_group]
    meta_group_b = meta_group_b[meta_group_b["genre"] != "None"]

    if selection_mode == "random":

        meta_group_b = meta_group_b.sample(frac=1)

    if selection_mode == "balanced":

        comparatives_group = np.unique(meta_group_b["genre"])

        i = 0
        g = 0
        indexes = []
        while i < len(meta_group_b):
            indexes.append(np.random.choice(list(meta_group_b[meta_group_b["genre"] == comparatives_group[g]].index), size=1))
            g += 1
            i += 1
            if g == len(comparatives_group):
                g = 0

        meta_group_b = meta_group_b.reindex([x for y in indexes for x in y])
    i = 0
    c = 1
    for index, row in meta_group_b.iterrows():

        dtm_path = path_to_seq_dtm + str(row["folder"]) + "/" + str(row["id"])
        data = pd.read_csv(dtm_path, sep="\t", dtype={"token": str, "count": np.int32})

        for seq_id in np.unique(data.iloc[:, 0]):

            seq = data[data.iloc[:, 0] == seq_id][["token", "count"]]
            seq.index = seq["token"]
            seq = seq.rename(columns={"count": "comp_" + str(i) + "_" + str(seq_id)})
            seq = seq.drop("token", axis=1)
            segments_group = pd.concat([segments_group, seq], axis=1, sort =False).fillna(0)

            if c >= segments_per_group:
                break

            c += 1
        i += 1
        if c >= segments_per_group:
            break
    segments_group = remove_stopwords(segments_group, stopwords)

    return segments_group


def unify_segments(seg_a, seg_b):

    full_data = pd.concat([seg_a, seg_b], axis=1, sort=False).fillna(0)
    seg_a = full_data[seg_a.columns]
    seg_b = full_data[seg_b.columns]

    return seg_a, seg_b


def bina(x):

    if x == 0:
        r = 0
    else:
        r = 1
    return r


def prepare_relativ(segment):

    segment["sum"] = segment.apply(lambda x: sum(x), axis=1)
    relative = segment["sum"] / sum(segment["sum"])
    docprops = segment["sum"].apply(lambda x: bina(x))

    return pd.Series(relative), pd.Series(docprops)


def prepare_formats(segments_1, segments_2):

    relfreqs1, docprops1 = prepare_relativ(segments_1)
    relfreqs2, docprops2 = prepare_relativ(segments_2)
    segments_1 = segments_1.drop("sum", axis=1)
    segments_2 = segments_2.drop("sum", axis=1)
    idlists = [list(segments_1.columns), list(segments_2.columns)]

    return relfreqs1, relfreqs2, docprops1, docprops2, idlists


def adjust_seglen(segment, s):

    cols = segment.columns
    i = 0

    while i <= len(cols) - s:
        segment[cols[i]] = np.array(segment.loc[:, cols[i:i + s]]).sum(axis=1)
        segment = segment.drop(cols[i + 1:i + s], axis=1)

        i += s

    return segment


# parameter
parser = argparse.ArgumentParser()
parser.add_argument('-target', default=None, type=str)
parser.add_argument('-path_to_seq_dtm', default=None, type=str)
parser.add_argument('-path_to_metadata', default=None, type=str)
parser.add_argument('-output_filepath', default=None, type=str)
parser.add_argument('-stopwords_file', default=None, type=str)
parser.add_argument('-selection_mode', default=None, type=str)
parser.add_argument('-segmentlen', default=None, type=int)
parser.add_argument('-max_segments_per_group', default=None, type=int)
parser.add_argument('-counter_size', default=None, type=str)
parser.add_argument('-logaddition', default=0.1, type=float)
args = parser.parse_args()

segmentlen = args.segmentlen
logaddition = args.logaddition
target = args.target
max_segments_per_group = args.max_segments_per_group

if max_segments_per_group == -1:
    max_segments_per_group = math.inf

path_to_seq_dtm = args.path_to_seq_dtm
path_to_metadata = args.path_to_metadata
output_filepath = args.output_filepath
stopwords_file = args.stopwords_file
selection_mode = args.selection_mode
counter_size = args.counter_size


# process
print("start")
meta = read_meta(path_to_metadata)
print("read_data ..")
segmentlen = int(segmentlen / 250)
segments_group_a = get_segments_focus_group(target, meta, max_segments_per_group, stopwords_file, segmentlen)
print("group_a:  " + str(len(segments_group_a.columns)) + " segments")

if counter_size == "full":

    counter_seg_size = math.inf

else:

    counter_seg_size = len(segments_group_a.columns)

segments_group_b = get_segments_comp_group(target, counter_seg_size, stopwords_file, selection_mode)
print("group_b:  " + str(len(segments_group_b.columns)) + " segments")

segments_group_a = adjust_seglen(segments_group_a, segmentlen)
segments_group_b = adjust_seglen(segments_group_b, segmentlen)
segments_group_a, segments_group_b = unify_segments(segments_group_a, segments_group_b)

print("transform ..")
relfreqs1, relfreqs2, docprops1, docprops2, idlists = prepare_formats(segments_group_a, segments_group_b)
print("calculate ..")
sd0, sd2, sr0, sr2, sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2, devprops1, devprops2 = calculate_scores(docprops1,
                                                                                                    docprops2,
                                                                                                    relfreqs1,
                                                                                                    relfreqs2,
                                                                                                    segments_group_a,
                                                                                                    segments_group_b,
                                                                                                    logaddition,
                                                                                                    args.segmentlen)

meanrelfreqs = get_meanrelfreqs(pd.concat([relfreqs1, relfreqs1], axis=1, sort=False))

res = combine_results(docprops1, docprops2, relfreqs1, relfreqs2, devprops1, devprops2, meanrelfreqs,
                      sd0, sd2, sr0, sr2, sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2)

output_filename = output_filepath + target + "_" + str(segmentlen * 250) + "_" + selection_mode + "_" + str(
    max_segments_per_group) + ".tsv"

save_results(res, output_filename)
print("write to: " + output_filename)
print("done.")