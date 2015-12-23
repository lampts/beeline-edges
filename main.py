#!/usr/bin/env python


import sys
import os.path
import json
import subprocess
from collections import namedtuple, Counter, defaultdict
from itertools import combinations, groupby, islice
from random import sample, random

import cjson
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc
# For cyrillic labels
rc('font', family='Verdana', weight='normal')

from scipy.sparse import coo_matrix
from sklearn.metrics import f1_score


TASK_DIR = 'task'
TRAIN = os.path.join(TASK_DIR, 'train.csv')
BEELINE_ID = 0
PARTITION = 'partition.json'
TOTAL_IDS = 1216082
SPLIT = 'split'
TABLES_DIR = 'tables'
TMP_TABLES_DIR = os.path.join(TABLES_DIR, 'tmp')
INTERSECTION_PARTS_TABLE = os.path.join(TABLES_DIR, 'intersection_parts.tsv')
INTERSECTIONS_TABLE = os.path.join(TABLES_DIR, 'intersections.tsv')
ID1_COLUMN = 0
ID2_COLUMN = 1
SUBMISSION = 'submission.csv'
DURATION_PARTS_TABLE = os.path.join(TABLES_DIR, 'duration_parts.tsv')
DURATIONS_TABLE = os.path.join(TABLES_DIR, 'durations.tsv')


TrainRecord = namedtuple(
    'TrainRecord',
    ['id1', 'id2',
     'id1_operator', 'id2_operator',
     'calls', 'duration', 'sms']
)


def load_train():
    table = pd.read_csv(TRAIN)
    for record in table.itertuples():
        index, A, B, x_A, x_B, c_AB, d_AB, c_BA, d_BA, s_AB, s_BA = record
        yield TrainRecord(A, B, x_A, x_B, c_AB, d_AB, s_AB)
        yield TrainRecord(B, A, x_B, x_A, c_BA, d_BA, s_BA)


def filter_active_edges(edges):
    for edge in edges:
        if edge.calls > 0 or edge.sms > 0:
            yield edge


def get_ids(train):
    return {id for record in train for id in (record.id1, record.id2)}


def get_id2s(train):
    return {record.id2 for record in train}


def get_edges(train):
    return {(record.id1, record.id2) for record in train}


def get_id_operator(train):
    id_operator = {}
    for record in train:
        id_operator[record.id1] = record.id1_operator
        id_operator[record.id2] = record.id2_operator
    return id_operator


def get_adjacency(train):
    adjacency = defaultdict(set)
    for record in train:
        id1 = record.id1
        id2 = record.id2
        adjacency[id1].add(id2)
        adjacency[id2].add(id1)
    return adjacency


def sample_connected_pairs(edges, size=10000):
    return sample(edges, size)


def sample_disconnected_pairs(ids, edges, size=10000):
    disconnected = zip(sample(ids, size), sample(ids, size))
    for id1, id2 in disconnected:
        assert id1 != id2
        assert (id1, id2) not in edges
    return disconnected


def show_contacts_intersection_difference(connected, disconnected, adjacency):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 4)

    data = Counter()
    for id1, id2 in connected:
        intersection = len(
            (adjacency[id1] - {id2})
            & (adjacency[id2] - {id1})
        )
        data[intersection] += 1
    table = pd.Series(data)
    table[:10].plot(ax=ax1)
    ax1.set_xlabel('contacts intersection')
    ax1.set_ylabel('# pairs')
    ax1.set_title('connected pairs')

    data = Counter()
    for id1, id2 in disconnected:
        intersection = len(adjacency[id1] & adjacency[id2])
        data[intersection] += 1
    for intersection in xrange(1, 10):
        data.setdefault(intersection, 0)
    table = pd.Series(data)
    table[:10].plot(ax=ax2)
    ax2.set_xlabel('contacts intersection')
    ax2.set_ylabel('# pairs')
    ax2.set_title('disconnected pairs')
    fig.tight_layout()


def show_operators_difference(connected, disconnected, id_operator):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(9, 4)

    for ax, pairs, title in (
            (ax1, connected, 'connected pairs'),
            (ax2, disconnected, 'disconnected pairs')
    ):
        data = Counter()
        for id1, id2 in pairs:
            data[id_operator[id1], id_operator[id2]] += 1
        table = pd.Series(data)
        table = table.unstack().fillna(0)
        sns.heatmap(table, ax=ax)
        ax.set_xlabel('id1 operator')
        ax.set_ylabel('id2 operator')
        ax.set_title(title)
    fig.tight_layout()


def split_train_test(train, train_share=0.8):
    subtrain = []
    test = []
    for record in train:
        if random() < train_share:
            subtrain.append(record)
        else:
            test.append(record)
    return subtrain, test


def get_coo_matrix(edges):
    ones = []
    id1s = []
    id2s = []
    for id1, id2 in edges:
        ones.append(1)
        id1s.append(id1)
        id2s.append(id2)
    return coo_matrix(
        (ones, (id1s, id2s)),
        shape=(TOTAL_IDS, TOTAL_IDS)
    )


def compute_f1(guess, etalon):
    return f1_score(
        get_coo_matrix(guess),
        get_coo_matrix(etalon),
        average='micro'
    )


def guess_random_edges(size=3):
    for id1 in xrange(TOTAL_IDS):
        for id2 in sample(xrange(TOTAL_IDS), size):
            yield id1, id2


def guess_random_edges_consider_existing(train):
    edges = get_edges(train)
    for edge in guess_random_edges():
        if edge not in edges:
            yield edge


def guess_random_edges_consider_existing_operator(train):
    edges = get_edges(train)
    for edge in guess_random_edges():
        if edge not in edges:
            yield edge


def guess_random_edges_consider_existing_operator(train):
    id_operator = get_id_operator(train)
    for id1, id2 in guess_random_edges_consider_existing(train):
        if ((id1 in id_operator and id_operator[id1] == BEELINE_ID)
            or (id2 in id_operator and id_operator[id2] == BEELINE_ID)):
            yield id1, id2


def log_progress(stream, every=1000, total=None):
    if total:
        every = total / 200     # every 0.5%
    for index, record in enumerate(stream):
        if index % every == 0:
            if total:
                progress = float(index) / total
                progress = '{0:0.2f}%'.format(progress * 100)
            else:
                progress = index
            print >>sys.stderr, progress,
        yield record


def filter_edges_by_calls(edges, min_calls=5):
    for edge in edges:
        if edge.calls >= min_calls:
            yield edge


def guess_like_directed_bechmark(train):
    adjacency = get_adjacency(
        filter_edges_by_calls(train, min_calls=5)
    )
    id_operator = get_id_operator(train)
    ids = xrange(TOTAL_IDS)
    for id1 in log_progress(ids, total=TOTAL_IDS):
        if id1 in adjacency:
            adjacent_ids = adjacency[id1]
            id2s = set()
            for adjacent_id in adjacent_ids:
                if adjacent_id in adjacency:
                    for id2 in adjacency[adjacent_id]:
                        if (id2 not in id2s
                            and id2 not in adjacent_ids
                            and id2 != id1
                            and ((id1 in id_operator
                                  and id_operator[id1] == BEELINE_ID)
                                 or (id2 in id_operator
                                     and id_operator[id2] == BEELINE_ID))):
                            yield id1, id2
                            id2s.add(id2)


def generate_intersection_parts(id2s, adjacency, max_id1s=100):
    for id2 in id2s:
        id1s = adjacency[id2]
        if len(id1s) < max_id1s:  # to boost up process a bit
            for source, target in combinations(id1s, 2):
                yield source, target


def serialize_intersection_parts(records):
    for id1, id2 in records:
        yield str(id1), str(id2)


def deserialize_intersection_parts(stream):
    for id1, id2 in stream:
        yield int(id1), int(id2)


def get_group_size(group):
    return sum(1 for _ in group)


def aggregate_intersection_parts(groups, min_intersection=2):
    for (id1, id2), group in groups:
        intersection = get_group_size(group)
        if intersection >= min_intersection:
            yield id1, id2, intersection


def serialize_intersections(records):
    for id1, id2, intersection in records:
        yield str(id1), str(id2), str(intersection)


def deserialize_intersections(stream):
    for id1, id2, intersection in stream:
        yield int(id1), int(id2), int(intersection)


def write_table(stream, table):
    with open(table, 'w') as file:
        file.writelines('\t'.join(_) + '\n' for _ in stream)


def read_table(table):
    with open(table) as file:
        for line in file:
            yield line.rstrip('\n').split('\t')


def get_table_size(table):
    output = subprocess.check_output(['wc', '-l', table])
    size, _ = output.split(None, 1)
    return int(size)


def sort_table(table, by, chunks=20):
    if not isinstance(by, (list, tuple)):
        by = (by,)
    size = get_table_size(table) / chunks
    tmp = os.path.join(TMP_TABLES_DIR, SPLIT)
    try:
        print >>sys.stderr, ('Split in {} chunks, prefix: {}'
                             .format(chunks, tmp))
        subprocess.check_call(
            ['split', '-l', str(size), table, tmp],
            env={'LC_ALL': 'C'}
        )
        ks = ['-k{0},{0}'.format(_ + 1) for _ in by]
        tmps = [os.path.join(TMP_TABLES_DIR, _)
                for _ in os.listdir(TMP_TABLES_DIR)]
        for index, chunk in enumerate(tmps):
            print >>sys.stderr, 'Sort {}/{}: {}'.format(
                index + 1, chunks, chunk
            )
            subprocess.check_call(
                ['sort', '-t', '\t'] + ks + ['-o', chunk, chunk],
                env={'LC_ALL': 'C'}
            )
        print >>sys.stderr, 'Merge into', table
        subprocess.check_call(
            ['sort', '-t', '\t'] + ks + ['-m'] + tmps + ['-o', table],
            env={'LC_ALL': 'C'}
        )
    finally:
        for name in os.listdir(TMP_TABLES_DIR):
            path = os.path.join(TMP_TABLES_DIR, name)
            os.remove(path)


def group_stream(stream, by):
    if isinstance(by, (list, tuple)):
        return groupby(stream, lambda r: [r[_] for _ in by])
    else:
        return groupby(stream, lambda r: r[by])


def load_edge_intersections(id_operator):
    records = deserialize_intersections(read_table(INTERSECTIONS_TABLE))
    intersections = Counter()
    for id1, id2, intersection in records:
        if (id1 != id2
            and (id_operator[id1] == BEELINE_ID
                 or id_operator[id2] == BEELINE_ID)):
            intersections[id1, id2] = intersection
    return intersections


def slice_edge_intersections(edge_intersections, existing, min_intersection=2):
    edges = set()
    for edge, intersection in edge_intersections.iteritems():
        if edge not in existing and intersection >= min_intersection:
            edges.add(edge)
    return edges


def make_edges_undirected(edges):
    undirected = set()
    for id1, id2 in edges:
        undirected.add((id1, id2))
        undirected.add((id2, id1))
    return undirected


def dump_submission(edges):
    adjacency = defaultdict(set)
    for id1, id2 in edges:
        adjacency[id1].add(id2)
    with open(SUBMISSION, 'w') as file:
        for id1 in adjacency:
            id2s = adjacency[id1]
            id2s = ','.join(str(_) for _ in id2s)
            line = '{id1},{id2s}\n'.format(
                id1=id1,
                id2s=id2s
            )
            file.write(line)


def get_edge_durations(train):
    durations = {}
    for record in train:
        durations[record.id1, record.id2] = record.duration
    return durations


def get_id21(train):
    id21 = defaultdict(set)
    for record in train:
        id21[record.id2].add(record.id1)
    return id21


def generate_duration_parts(id21, durations, max_id1s=100):
    for id2, id1s in log_progress(id21.iteritems(), total=len(id21)):
        if len(id1s) < max_id1s:  # to boost up process a bit
            for source, target in combinations(id1s, 2):
                duration = durations[source, id2] + durations[target, id2]
                # make edges unordered right away
                if source > target:
                    source, target = target, source
                yield source, target, duration



def serialize_duration_parts(records):
    for id1, id2, duration in records:
        yield str(id1), str(id2), str(duration)


def deserialize_duration_parts(stream):
    for id1, id2, duration in stream:
        yield int(id1), int(id2), int(duration)


def aggregate_duration_parts(groups, min_duration=2):
    for (id1, id2), group in groups:
        duration = sum(part for _, _, part in group)
        if duration >= min_duration:
            yield id1, id2, duration


serialize_durations = serialize_duration_parts
deserialize_durations = deserialize_duration_parts


def show_durations_histogram(durations_histogram):
    table = pd.Series(durations_histogram)
    total = table.sum()
    limit = 4000
    top = table[table.index > limit].sum()
    print limit, ':', top, '/', total
    fig, ax = plt.subplots()
    table[:1000].plot(ax=ax)
    ax.set_xlabel('duration')
    ax.set_ylabel('# edges')


def load_edge_durations(id_operator, min_duration=4000):
    durations_size = get_table_size(DURATIONS_TABLE)
    records = deserialize_durations(read_table(DURATIONS_TABLE))
    records = log_progress(records, total=durations_size)
    durations = Counter()
    for id1, id2, duration in records:
        if (duration >= min_duration
            and id1 != id2
            and (id_operator[id1] == BEELINE_ID
                 or id_operator[id2] == BEELINE_ID)):
            durations[id1, id2] = duration
    return durations


def slice_edge_durations(edge_durations, existing, min_duration=10000):
    edges = set()
    for edge, duration in edge_durations.iteritems():
        if edge not in existing and duration >= min_duration:
            edges.add(edge)
    return edges
