# author: TangChen
# email: chen_tang1999@163.com
# date: 

'''
GitHub: https://github.com/Corgiperson, welcome to star my project!
description: 
'''
# -*- coding: UTF-8 -*-
# Python version: 3.6
import os as _os
import numpy as _np
import nmslib as _nmslib
from tqdm import tqdm as _tqdm
from imagededup.methods import CNN as _CNN
class Hash():
    def __init__(self, algorithm='imagededup:cnn', hashobj=None):
        """ Generate Hash for image data and search similiar data
        :param algorithm: The hash algorithm, current options are `imagededup:cnn`
        :param hashobj: The Hash object
        """
        self.__ALGORITHM_SET = {'imagededup:cnn'}
        if algorithm not in self.__ALGORITHM_SET:
            raise ValueError(
                "Hash algorithm must be in {0}".format(self.__ALGORITHM_SET))
        self.__algorithm = algorithm
        if self.__algorithm == 'imagededup:cnn':
            self.__hashalgo = _CNN()
        else:
            pass
        self.__hashobj = hashobj
    def hashFile(self, filepath):
        """ Generate image file Hash
        :param filepath: The path to the image file
        :return: Hash obj
        """
        if not _os.path.isfile(filepath):
            raise ValueError('Input path is not a file')
        if self.__algorithm == 'imagededup:cnn':
            encoding = self.__hashalgo.encode_image(image_file=filepath)
            if encoding is None:
                raise ValueError('Input image was corrupted.')
            self.__hashobj = encoding[0]
            return encoding[0]
        else:
            pass
    def hashFiles(self, filedir, min_similarity_threshold=0.85):
        """ Generate image file Hash
        :param filepath: The path to the image files
        :param min_similarity_threshold: Threshold
        :return: Hash obj
        """
        if not _os.path.isdir(filedir):
            raise ValueError('Input path is not a directory')
        if self.__algorithm == 'imagededup:cnn':
            encodings = self.__hashalgo.encode_images(filedir)
            duplicates = self.__hashalgo.find_duplicates(
                encoding_map=encodings, min_similarity_threshold=min_similarity_threshold, scores=True)
            return encodings, duplicates
        else:
            pass
    def getHexHashString(self):
        """ Get hex value of Hash
        :return: string
        """
        if self.__hashobj is None:
            raise ValueError(
                "Please use Hash.hashFile(filepath) method first")
        if self.__algorithm == 'imagededup:cnn':
            return self.__hashobj.tobytes().hex()
        else:
            pass
    def searchNN(self, index_obj, nn=20, ef=36):
        """ kNN search
        :param index_obj: The index object of database
        :param nn: The number of nearest neighbor
        :param ef: The larger value of `ef`, the more candidates to search,
                   the larger value of recall, the slower speed to search
        :return: List [ids, distances]
        """
        if self.__hashobj is None:
            raise ValueError(
                "Please use Hash.hashFile(filepath) method first")
        if self.__algorithm == 'imagededup:cnn':
            index_obj.setQueryTimeParams({'efSearch': ef})
            nids, ndist = index_obj.knnQuery(self.__hashobj, k=nn)
            return nids, 1-ndist
        else:
            pass
def doIndexHnswCosine(filelist, M=30, efC=100, num_threads=1, save_data=False):
    """ Do indexing job for image database
    :param stringlist: Dict obj, with {index id --> int: image filepath, ...} format
    :param M: is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M).
              Higher M leads to higher accuracy/run_time at fixed ef/efConstruction.
    :param efC: Controls index search speed/build speed tradeoff
    :param num_threads: The number of thread used in CPU
    :param save_data: Whether save all return data or not
    :return: (cosine data, id data, index obj)
    """
    index = _nmslib.init(method='hnsw', space='cosinesimil')
    index_time_params = {
        'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
    hash = Hash()
    train, id = [], []
    for i in _tqdm(filelist, ascii=True, desc="Hashing..."):
        m = hash.hashFile(filelist[i])
        id.append(i)
        train.append(m)
    index.addDataPointBatch(train, id)
    index.createIndex(index_time_params, print_progress=True)
    if save_data:
        _np.save('cosine_data.npy', train, allow_pickle=True)
        _np.save('id_data.npy', id, allow_pickle=True)
        index.saveIndex('index.bin', save_data=False)
    return train, id, index
def loadHnswCosineIndex(cosine_data_path, id_data_path, index_bin_path):
    """ Load hnsw index from saved index file
    :param cosine_data_path: numpy file path
    :param id_data_path: id file path
    :param index_bin_path: index binary file path
    :return: index obj
    """
    _train = _np.load(cosine_data_path, allow_pickle=True)
    _id = _np.load(id_data_path, allow_pickle=True)
    index = _nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(_train, _id)
    index.loadIndex(index_bin_path, load_data=False)
return index
# -*- coding: UTF-8 -*-
# Python version: 3.6
import os as _os
import csv as _csv
from tqdm import tqdm as _tqdm
from datasketch import MinHash as _MinHash
from datasketch import MinHashLSH as _MinHashLSH
from itertools import combinations as _combinations
from datasketch import MinHashLSHForest as _MinHashLSHForest
def _readcsv(filepath):
    """ Normal read csv file, ignoring difference between header and content
    读CSV文件并返回二位字符串数组，不区分表格头和表格内容
    :param filepath: CSV File path
    :return: List of list of string
    """
    table = []
    with open(filepath, 'r') as csvfile:
        spamreader = _csv.reader(csvfile)
        for row in spamreader:
            table.append(row)
    return table
class Hash():
    def __init__(self, algorithm='datasketch:minhashforest', readfunc=_readcsv, hashobj=None, ncombination=3):
        """ Generate Hash for CSV data and search similiar data
        :param algorithm: The hash algorithm, current options are `datasketch:minhashlsh`, `datasketch:minhashforest`, `hnsw:xxx`
        :param readfunc: The function that read the CSV file and return one string
        :param hashobj: The Hash object
        :param ncombination: The combination value `n` of csv, 3 is efficient, fast
        """
        self.__ALGORITHM_SET = {'datasketch:minhashlsh',
                                'datasketch:minhashforest', 'hnsw:xxx'}
        self.__algorithm = algorithm
        if algorithm not in self.__ALGORITHM_SET:
            raise ValueError(
                "Hash algorithm must be in {0}".format(self.__ALGORITHM_SET))
        if not callable(readfunc):
            raise ValueError("The readfunc must be a callable.")
        self.__readfunc = readfunc
        self.__ncombination = ncombination
        self.__hashobj = hashobj
    def hashFile(self, filepath):
        """ Generate CSV file Hash
        :param filepath: The path to the CSV file
        :return: Hash obj
        """
        if not _os.path.isfile(filepath):
            raise ValueError('Input path is not a file')
        content = self.__readfunc(filepath)
        if self.__algorithm in {'datasketch:minhashlsh', 'datasketch:minhashforest'}:
            m = _MinHash()
            for i in content:
                for j in _combinations(i, self.__ncombination):
                    m.update(''.join(j).encode())
            self.__hashobj = m
            return m
        elif self.__algorithm == 'hnsw:xxx':
            pass
    def getHexHashString(self):
        """ Get hex value of Hash
        :return: string
        """
        if self.__hashobj is None:
            raise ValueError(
                "Please use Hash.hashFile(filepath) method first")
        if self.__algorithm in {'datasketch:minhashlsh', 'datasketch:minhashforest'}:
            return self.__hashobj.hashvalues.tobytes().hex()
        elif self.__algorithm == 'hnsw:xxx':
            pass
    def searchNN(self, index_obj, nn=20):
        """ kNN search
        :param index_obj: The index object of database
        :param nn: The number of nearest neighbor
        :return: List [ids]
        """
        if self.__hashobj is None:
            raise ValueError(
                "Please use Hash.hashFile(filepath) method first")
        if self.__algorithm == 'datasketch:minhashforest':
            return index_obj.query(self.__hashobj, nn)
        elif self.__algorithm == 'datasketch:minhashlsh':
            return index_obj.query(self.__hashobj)
        elif self.__algorithm == 'hnsw:xxx':
            pass
def doIndexMinHashForest(filelist):
    """ Do indexing job for CSV database
    :param filelist: Dict obj, with {'index_name': CSV filepath, ...} format
    :return: forest obj
    """
    forest = _MinHashLSHForest()
    minhash = Hash()
    for i in _tqdm(filelist, ascii=True, desc="Hashing..."):
        m = minhash.hashFile(filelist[i])
        forest.add(i, m)
    print('Indexing...')
    forest.index()
    return forest
def doIndexMinHashLSH(filelist, threshold=0.5):
    """ Do indexing job for CSV database
    :param filelist: Dict obj, with {'index_name': CSV filepath, ...} format
    :param threshold: Similirity threshold
    :return: (threshold, lsh obj)
    """
    lsh = _MinHashLSH(threshold=threshold)
    minhash = Hash()
    for i in _tqdm(filelist, ascii=True, desc="Hashing..."):
        m = minhash.hashFile(filelist[i])
        lsh.insert(i, m)
return (threshold, lsh)
# -*- coding: UTF-8 -*-
# Python version: 3.6
import os as _os
import re as _re
import nltk as _nltk
from tqdm import tqdm as _tqdm
from datasketch import MinHash as _MinHash
from datasketch import MinHashLSHForest as _MinHashLSHForest
from datasketch import MinHashLSH as _MinHashLSH
def cleanhtml(raw_html):
    cleanr = _re.compile('<.*?>')
    cleantext = _re.sub(cleanr, '', raw_html)
    return cleantext
def join_punctuation(seq, characters='.,;:?!\''):
    """ Merge sentences more in line with English grammar
        因为分词将标点符号分为单独的单词，本函数使合并的时候不添加单独的空格
    :param seq: word list with `list` format
    :param characters: Collection of punctuation to consider
    :return: list, Modified word list
    """
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)
    for nxt in seq:
        if nxt[0] in characters:
            current += nxt
        else:
            yield current
            current = nxt
    yield current
def _readtxt(filepath):
    """ Normal read text method, clear html tag
    :param filepath: File path
    :return: string
    """
    with open(filepath, 'r') as f:
        data = f.readlines()
    txt = ''.join(data)
    return cleanhtml(txt)
class Hash():
    def __init__(self, algorithm='datasketch:minhashforest', readfunc=_readtxt, hashobj=None, ngram=3):
        """ Generate Hash for text data and search similiar data
        :param algorithm: The hash algorithm, current options are `datasketch:minhashlsh`, `datasketch:minhashforest`, `hnsw:xxx`
        :param readfunc: The function that read the text file and return one string
        :param hashobj: The Hash object
        :param ngram: The value `n` in n-gram, 3 is suitable for short text
        """
        self.__ALGORITHM_SET = {'datasketch:minhashlsh',
                                'datasketch:minhashforest', 'hnsw:xxx'}
        self.__algorithm = algorithm
        if algorithm not in self.__ALGORITHM_SET:
            raise ValueError(
                "Hash algorithm must be in {0}".format(self.__ALGORITHM_SET))
        if not callable(readfunc):
            raise ValueError("The readfunc must be a callable.")
        self.__readfunc = readfunc
        self.__ngram = ngram
        self.__hashobj = hashobj
    def hashString(self, string):
        """ Generate string Hash
        :param string: Text string
        :return: Hash obj
        """
        if self.__algorithm in {'datasketch:minhashlsh', 'datasketch:minhashforest'}:
            m = _MinHash()
            for i in _nltk.ngrams(string, self.__ngram):
                m.update(''.join(i).encode())
            self.__hashobj = m
            return m
        elif self.__algorithm == 'hnsw:xxx':
            pass
    def hashFile(self, filepath):
        """ Generate text file Hash
        :param filepath: The path to the text file
        :return: Hash obj
        """
        if not _os.path.isfile(filepath):
            raise ValueError('Input path is not a file')
        content = self.__readfunc(filepath)
        return self.hashString(content)
    def getHexHashString(self):
        """ Get hex value of Hash
        :return: string
        """
        if self.__hashobj is None:
            raise ValueError(
                "Please use Hash.hashFile(filepath) method first")
        if self.__algorithm in {'datasketch:minhashlsh', 'datasketch:minhashforest'}:
            return self.__hashobj.hashvalues.tobytes().hex()
        elif self.__algorithm == 'hnsw:xxx':
            pass
    def searchNN(self, index_obj, nn=20):
        """ kNN search
        :param index_obj: The index object of database
        :param nn: The number of nearest neighbor
        :return: List [ids]
        """
        if self.__hashobj is None:
            raise ValueError(
                "Please use Hash.hashFile(filepath) method first")
        if self.__algorithm == 'datasketch:minhashforest':
            return index_obj.query(self.__hashobj, nn)
        elif self.__algorithm == 'datasketch:minhashlsh':
            return index_obj.query(self.__hashobj)
        elif self.__algorithm == 'hnsw:xxx':
            pass
def doIndexMinHashForest(stringlist):
    """ Do indexing job for text database
    :param stringlist: Dict obj, with {'index_name': text string, ...} format
    :return: forest obj
    """
    forest = _MinHashLSHForest()
    minhash = Hash()
    for i in _tqdm(stringlist, ascii=True, desc="Hashing..."):
        m = minhash.hashString(stringlist[i])
        forest.add(i, m)
    print('Indexing...')
    forest.index()
    return forest
def doIndexMinHashLSH(stringlist, threshold=0.5):
    """ Do indexing job for text database
    :param stringlist: Dict obj, with {'index_name': text string, ...} format
    :param threshold: Similirity threshold
    :return: (threshold, lsh obj)
    """
    lsh = _MinHashLSH(threshold=threshold)
    minhash = Hash()
    for i in _tqdm(stringlist, ascii=True, desc="Hashing..."):
        m = minhash.hashString(stringlist[i])
        lsh.insert(i, m)
return (threshold, lsh)
# -*- coding: UTF-8 -*-
# Python version: 3.6
import os as _os
import numpy as _np
from tqdm import tqdm as _tqdm
from .visil.model import ViSiL as _ViSil
from .visil.utils import load_video as _load_video
class Hash():
    def __init__(self, model_dir, algorithm='cnn:visil', readfunc=_load_video, hashobj=None, gpu_id=0):
        """ Generate Hash for video data and search similiar data
        :param model_dir: Model directory for loading model parameters
        :param algorithm: The hash algorithm, current options are `cnn:visil`
        :param readfunc: The function to read video into matrix
        :param hashobj: The Hash object
        :gpu_id: GPU id
        """
        self.__ALGORITHM_SET = {'cnn:visil'}
        if algorithm not in self.__ALGORITHM_SET:
            raise ValueError(
                "Hash algorithm must be in {0}".format(self.__ALGORITHM_SET))
        self.__algorithm = algorithm
        if self.__algorithm == 'cnn:visil':
            self.__algomodel = _ViSiL(
                model_dir, load_queries=False, gpu_id=gpu_id, queries_number=None)
        else:
            pass
        if not callable(readfunc):
            raise ValueError("The readfunc must be a callable.")
        self.__readfunc = readfunc
        self.__hashobj = hashobj
    def hashFile(self, filepath, batch_size=100):
        """ Generate video file Hash
        :param filepath: The path to the video file
        :param batch_size: Batch size
        :return: Hash obj
        """
        if not _os.path.isfile(filepath):
            raise ValueError('Input path is not a file')
        data = self.__readfunc(filepath)
        if self.__algorithm == 'cnn:visil':
            encoding = self.__algomodel.extract_features(data, batch_size)
            self.__hashobj = encoding
            return encoding
        else:
            pass
    def getHexHashString(self):
        """ Get hex value of Hash
        :return: string
        """
        if self.__hashobj is None:
            raise ValueError(
                "Please use Hash.hashFile(filepath) method first")
        if self.__algorithm == 'cnn:visil':
            return self.__hashobj.tobytes().hex()
        else:
            pass
    def calculateSimilarities(self, candidate_list, batch_size=100):
        """ Calculate similarities for each item in candidate_list
        :param candidate_list: List of feature/hash for videos
        :param batch_size: Batch size
        :return: Similarities list ranged [-1, 1]
        """
        if self.__hashobj is None:
            raise ValueError(
                "Please use Hash.hashFile(filepath) method first")
        if self.__algorithm == 'cnn:visil':
            self.__algomodel.set_queries(candidate_list)
            sims = self.__algomodel.calculate_similarities(
                self.__hashobj, batch_size)
            return sims
        else:
            pass
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import csv
import time
import gzip
import click
import pickle
import nmslib
import hashlib
import zipfile
import numpy as np
from gexf import Gexf
from scipy import spatial
from datetime import datetime
from lfp import fpText, fpImage, fpTable
from flask import Flask, render_template, request, redirect, url_for, session, escape, flash, send_from_directory

def readPkl(src):
    with gzip.open(src, 'rb') as f:
        data = pickle.load(f)
    return data
def writePkl(src, obj):
    with gzip.open(src, 'wb') as f:
        pickle.dump(file=f, obj=obj)
def readcsv(filepath):
    """ Normal read csv file, ignoring difference between header and content
    :param filepath: CSV File path
    :return: string
    """
    table = []
    with open(filepath, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            table.append(' | '.join(row))
    return '\n'.join(table)
def readtxt(filepath):
    """ Normal read text method, clear html tag
    :param filepath: File path
    :return: string
    """
    with open(filepath, 'r') as f:
        data = f.readlines()
    txt = ''.join(data)
    return fpText.cleanhtml(txt)
# 准备文本相关
print(' -*- loading txt index...')
txt_id_index = readPkl(
    './static/dataset_resource/lmrd_id_index_lshforest.pkl.gz')
txt_id_string = readPkl(
    './static/dataset_resource/lmrd_id_string_minhash_lshfroest.pkl.gz')
h_txt = fpText.Hash()
# 准备图片相关
print(' -*- loading img index...')
_train = np.load('./static/dataset_resource/flickr_cosine_data_v2.npy', allow_pickle=True)
#  _train = np.load('./static/dataset_resource/flickr_cosine_data.npy', allow_pickle=True)
img_index = nmslib.init(method='hnsw', space='cosinesimil')
_id = list(range(1000012))
#  _id = list(range(1000000))
_id.remove(52478)
_id.remove(100150)
_id.remove(100459)
_id.remove(108511)
_id.remove(682854)
#  _id.remove(59898)
#  _id.remove(107349)
#  _id.remove(104442)
#  _id.remove(108460)
#  _id.remove(686806)
img_index.addDataPointBatch(_train, _id)
img_index.loadIndex('./static/dataset_resource/flickr_index_v2.bin', load_data=False)
img_id_path = readPkl('./static/dataset_resource/flickr_id_imagepath_v2.pkl.gz')
#  img_index.loadIndex('./static/dataset_resource/flickr_index.bin', load_data=False)
#  img_id_path = readPkl('./static/dataset_resource/flickr_id_imagepath.pkl.gz')
h_img = fpImage.Hash()
# 准备表格相关
print(' -*- loading csv index...')
tab_path_index = readPkl('./static/dataset_resource/table_path_index.pkl.gz')
h_tab = fpTable.Hash()
print(' -*- loading over...')
app = Flask(__name__)
app.config['UPLOADED_PATH'] = os.path.join(app.root_path, 'static/upload')
app.config['GRAPH_PATH'] = os.path.join(app.root_path, 'static/graph_resource')
app.secret_key = '237e99220ee13109de9ea37d462d91d15c88c734d6ddb0e9dd1664562c430af5'
ALLOWED_TEXT = set(['txt', ])
ALLOWED_IMAGE = set(['png', 'jpg', 'jpeg', 'JPG', 'JPEG', ])
ALLOWED_TABLE = set(['csv', ])
ALLOWED_ZIP = set(['zip', ])
ALLOWED_EXTENSIONS = set([*ALLOWED_TEXT, *ALLOWED_IMAGE, *ALLOWED_TABLE, *ALLOWED_ZIP, 'mp4', 'mp3'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
def is_zipfil_one_extension(filelist):
    s = set()
    for i in filelist:
        if not '.' in i:
            return False, u'ERROR: 存在不合法的文件拓展名'
        ext = i.rsplit('.', 1)[1]
        if not ext in ALLOWED_EXTENSIONS:
            return False, u'存在不支持处理的文件类型'
        s.add(ext)
    if(len(s) > 1):
        return False, u'ERROR: 不支持多种类型文件同时处理'
    return True, s.pop()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        for f in request.files.getlist('file'):
            if allowed_file(f.filename):
                extension = f.filename.rsplit('.', 1)[1]
                hashed_filename = '{0}.{1}'.format(hashlib.md5('{0}{1}'.format(
                    datetime.now(), f.filename).encode()).hexdigest(), extension)
                f.save(os.path.join(
                    app.config['UPLOADED_PATH'], hashed_filename))
                session['filename'] = hashed_filename
                session['extension'] = extension
            else:  # ??? not work
                flash(u'该格式文件尚未支持，请静待更新')
                return redirect(url_for('index'))
            print(' -*- uploading:', session)
            break
    return render_template('index.html')
def hash_one_file(hash_class, filepath, hash_index):
    t0 = time.time()
    encoding = hash_class.hashFile(filepath)
    time_hash = time.time()-t0  # RETURN 生成哈希的时间
    encodingstring = hash_class.getHexHashString()  # RETURN 哈希的字符串形式
    t0 = time.time()
    nn = hash_class.searchNN(hash_index)
    time_search = time.time()-t0  # RETURN 检索的时间
    return time_hash, time_search, encoding, encodingstring, nn
@app.route('/run')
def runrun():
    print(' -*- running:', session)
    if 'filename' not in session:
        flash(u'请您先上传一个支持处理的文件或者等待文件上传成功')
        return redirect(url_for('index'))
    elif not os.path.isfile(os.path.join(app.config['UPLOADED_PATH'], session['filename'])):
        flash(u'请您等待文件上传成功')
        session.pop('filename', None)
        session.pop('extension', None)
        return redirect(url_for('index'))
    filepath = os.path.join(app.config['UPLOADED_PATH'], session['filename'])
    file_extensions = filepath.rsplit('.', 1)[1]
    session_filename = session['filename']
    session_extension = session['extension']
    session.pop('filename', None)
    session.pop('extension', None)
    if file_extensions in ALLOWED_TEXT:  # 文本类型文件
        try:
            time_hash, time_search, encoding, encodingstring, nn = hash_one_file(
                h_txt, filepath, txt_id_index)
            query_result = []
            for i in nn:
                query_result.append((i, encoding.jaccard(txt_id_string[i][1])))
            query_result = sorted(
                query_result, key=lambda x: x[1], reverse=True)
            query_txt = readtxt(filepath)  # RETURN 被检索的文件内容
            hashobj_path = filepath+'.hashobj'  # RETURN 哈希格式下载
            writePkl(hashobj_path, encoding)
            hashobj_path = '/download/'+session_filename+'.hashobj'
        except Exception as e:
            flash(u'处理中发生错误{0}，请重新提交'.format(e))
            return redirect(url_for('index'))
        txt_list = []  # RETURN 检索结果排序
        for i, j in enumerate(query_result):
            txt_list.append((i, round(j[1]*100, 2), txt_id_string[j[0]][0]))
        first_txt = txt_list[0][2]  # RETURN 排名第一的文本的内容
        num = len([i for i in txt_list if i[1] > 80])
        if num:
            info = ['找到{0}个疑似平台泄漏的数据'.format(num), 'error']
        else:
            info = ['未发现疑似平台泄漏的数据', 'success']
        return render_template('txt-result.html', time_hash=time_hash, time_search=time_search, encodingstring=encodingstring, query_txt=query_txt, hashobj_path=hashobj_path, txt_list=txt_list, info=info, first_txt=first_txt)
    elif file_extensions in ALLOWED_IMAGE:  # 图像类型文件
        try:
            time_hash, time_search, encoding, encodingstring, nn = hash_one_file(
                h_img, filepath, img_index)
            query_result = []  # RETURN 检索结果排序
            for i in range(len(nn[0])):
                query_result.append(
                    (i, round(nn[1][i]*100, 2), img_id_path[nn[0][i]]))
            query_img_url = os.path.join(
                'static/upload', session_filename)  # RETURN 被检索的文件内容
            hashobj_path = filepath+'.hashobj'  # RETURN 哈希格式下载
            writePkl(hashobj_path, encoding)
            hashobj_path = '/download/'+session_filename+'.hashobj'
        except Exception as e:
            flash(u'处理中发生错误{0}，请重新提交'.format(e))
            return redirect(url_for('index'))
        num = len([i for i in query_result if i[1] > 85])
        if num:
            info = ['找到{0}个疑似平台泄漏的数据'.format(num), 'error']
        else:
            info = ['未发现疑似平台泄漏的数据', 'success']
        return render_template('img-result.html', time_hash=time_hash, time_search=time_search, encodingstring=encodingstring, query_img_url=query_img_url, hashobj_path=hashobj_path, info=info, img_list=query_result)
    elif file_extensions in ALLOWED_TABLE:  # 表格类型文件
        try:
            time_hash, time_search, encoding, encodingstring, nn = hash_one_file(
                h_tab, filepath, tab_path_index)
            query_result = []
            for i in nn:
                query_result.append((i, encoding.jaccard(h_tab.hashFile(i))))
            query_result = sorted(
                query_result, key=lambda x: x[1], reverse=True)
            query_tab = readcsv(filepath)  # RETURN 被检索的文件内容
            hashobj_path = filepath+'.hashobj'  # RETURN 哈希格式下载
            writePkl(hashobj_path, encoding)
            hashobj_path = '/download/'+session_filename+'.hashobj'
        except Exception as e:
            flash(u'处理中发生错误{0}，请重新提交'.format(e))
            return redirect(url_for('index'))
        tab_list = []  # RETURN 检索结果排序
        for i, j in enumerate(query_result):
            tab_list.append((i, round(j[1]*100, 2), readcsv(j[0])))
        first_tab = tab_list[0][2]  # RETURN 排名第一的表格的内容
        num = len([i for i in tab_list if i[1] > 70])
        if num:
            info = ['找到{0}个疑似平台泄漏的数据'.format(num), 'error']
        else:
            info = ['未发现疑似平台泄漏的数据', 'success']
        return render_template('tab-result.html', time_hash=time_hash, time_search=time_search, encodingstring=encodingstring, query_tab=query_tab, hashobj_path=hashobj_path, tab_list=tab_list, info=info, first_tab=first_tab)
    elif file_extensions in ALLOWED_ZIP:  # 同类型多个文件
        # 解压部分
        r = zipfile.is_zipfile(filepath)
        if not r:
            flash(u'输入文件无法解压，请重新提交')
            return redirect(url_for('index'))
        else:
            with zipfile.ZipFile(filepath, 'r') as zf:
                zipnamelist = zf.namelist()
            stat, msg = is_zipfil_one_extension(zipnamelist)  # 判断是否是同类型文件
            if not stat:
                flash(msg)
                return redirect(url_for('index'))
            desc_dir = filepath.rsplit('.', 1)[0]  # 解压目标目录
            with zipfile.ZipFile(filepath, 'r') as zf:
                for file in zipnamelist:
                    zf.extract(file, desc_dir)
                    os.rename(os.path.join(desc_dir, file), os.path.join(
                        desc_dir, file.replace('-', '_')))
        zipnamelist = [i.replace('-', '_') for i in zipnamelist]
        # 开始处理
        file_count = len(zipnamelist)
        if msg in ALLOWED_TEXT:
            time_cal_list = []
            hash_code_list = {}
            dict_filename_id = {}
            for cnt, i in enumerate(os.listdir(desc_dir)):
                t0 = time.time()
                encoding = h_txt.hashFile(os.path.join(desc_dir, i))
                time_cal_list.append(time.time()-t0)
                encodingstring = h_txt.getHexHashString()
                html_id = hashlib.md5('{0}{1}'.format(
                    time.time(), i).encode()).hexdigest()
                tmp_hash_file = '{0}.{1}.hashobj'.format(html_id, msg)
                hashobj_path = os.path.join(
                    app.config['UPLOADED_PATH'], tmp_hash_file)  # RETURN 哈希格式下载
                writePkl(hashobj_path, encoding)
                hashobj_path = '/download/'+tmp_hash_file
                hash_code_list[i] = (str(cnt), html_id, encoding,
                                     encodingstring, hashobj_path)
                dict_filename_id[i] = str(cnt)
            file_avg_time = sum(time_cal_list)/len(time_cal_list)
            dict_simi_pair = {i: [] for i in hash_code_list}
            dict_simi_pair_info = {i: [] for i in hash_code_list}
            for i in hash_code_list:
                for j in hash_code_list:
                    if i != j:
                        tmp_h = hash_code_list[j]
                        tmp_simi = hash_code_list[i][2].jaccard(tmp_h[2])
                        if tmp_simi > 0.8:
                            dict_simi_pair[i].append(tmp_h[0])
                            dict_simi_pair_info[i].append((j, tmp_simi))
            # 画图
            gexf_filename = session_filename[:-4]
            gexf = Gexf("Charles", "Network Structure")
            graph = gexf.addGraph("undirected", "static", "Network Sturcture")
            attr = graph.addNodeAttribute(
                'Modularity Class', '1', force_id='modularity_class')
            for i in hash_code_list:
                tmp_h = hash_code_list[i]
                node = graph.addNode(
                    tmp_h[0], i+'-'+tmp_h[0]+'-'+tmp_h[1], r='255', g='255', b='255', size='30')
                node.addAttribute(attr, '0')
            graph_edge_set = set()
            for i in dict_simi_pair:
                for j in dict_simi_pair[i]:
                    tmp = (dict_filename_id[i], j)
                    if (tmp[1], tmp[0]) in graph_edge_set or tmp in graph_edge_set:
                        pass
                    else:
                        graph_edge_set.add(tmp)
            edgecnt = 0
            for i in graph_edge_set:
                graph.addEdge(edgecnt, i[0], i[1], weight='1')
                edgecnt += 1
            # 输出gexf图
            outfile = os.path.join(
                app.config['GRAPH_PATH'], '{0}.gexf'.format(gexf_filename))
            with open(outfile, 'wb') as f:
                gexf.write(f)
            return render_template('zip-txt-result.html', gexf_filename=gexf_filename, file_count=file_count, file_avg_time=file_avg_time, hash_code_list=hash_code_list, dict_simi_pair_info=dict_simi_pair_info)
        elif msg in ALLOWED_TABLE:
            time_cal_list = []
            hash_code_list = {}
            dict_filename_id = {}
            for cnt, i in enumerate(os.listdir(desc_dir)):
                t0 = time.time()
                encoding = h_tab.hashFile(os.path.join(desc_dir, i))
                time_cal_list.append(time.time()-t0)
                encodingstring = h_tab.getHexHashString()
                html_id = hashlib.md5('{0}{1}'.format(
                    time.time(), i).encode()).hexdigest()
                tmp_hash_file = '{0}.{1}.hashobj'.format(html_id, msg)
                hashobj_path = os.path.join(
                    app.config['UPLOADED_PATH'], tmp_hash_file)  # RETURN 哈希格式下载
                writePkl(hashobj_path, encoding)
                hashobj_path = '/download/'+tmp_hash_file
                hash_code_list[i] = (str(cnt), html_id, encoding,
                                     encodingstring, hashobj_path)
                dict_filename_id[i] = str(cnt)
            file_avg_time = sum(time_cal_list)/len(time_cal_list)
            dict_simi_pair = {i: [] for i in hash_code_list}
            dict_simi_pair_info = {i: [] for i in hash_code_list}
            for i in hash_code_list:
                for j in hash_code_list:
                    if i != j:
                        tmp_h = hash_code_list[j]
                        tmp_simi = hash_code_list[i][2].jaccard(tmp_h[2])
                        if tmp_simi > 0.7:
                            dict_simi_pair[i].append(tmp_h[0])
                            dict_simi_pair_info[i].append((j, tmp_simi))
            # 画图
            gexf_filename = session_filename[:-4]
            gexf = Gexf("Charles", "Network Structure")
            graph = gexf.addGraph("undirected", "static", "Network Sturcture")
            attr = graph.addNodeAttribute(
                'Modularity Class', '1', force_id='modularity_class')
            for i in hash_code_list:
                tmp_h = hash_code_list[i]
                node = graph.addNode(
                    tmp_h[0], i+'-'+tmp_h[0]+'-'+tmp_h[1], r='255', g='255', b='255', size='30')
                node.addAttribute(attr, '0')
            graph_edge_set = set()
            for i in dict_simi_pair:
                for j in dict_simi_pair[i]:
                    tmp = (dict_filename_id[i], j)
                    if (tmp[1], tmp[0]) in graph_edge_set or tmp in graph_edge_set:
                        pass
                    else:
                        graph_edge_set.add(tmp)
            edgecnt = 0
            for i in graph_edge_set:
                graph.addEdge(edgecnt, i[0], i[1], weight='1')
                edgecnt += 1
            # 输出gexf图
            outfile = os.path.join(
                app.config['GRAPH_PATH'], '{0}.gexf'.format(gexf_filename))
            with open(outfile, 'wb') as f:
                gexf.write(f)
            return render_template('zip-tab-result.html', gexf_filename=gexf_filename, file_count=file_count, file_avg_time=file_avg_time, hash_code_list=hash_code_list, dict_simi_pair_info=dict_simi_pair_info)
        elif msg in ALLOWED_IMAGE:
            time_cal_list = []
            hash_code_list = {}
            dict_filename_id = {}
            icon_path_prefix = 'static/upload/'+desc_dir.rsplit('/', 1)[1]
            for cnt, i in enumerate(os.listdir(desc_dir)):
                t0 = time.time()
                encoding = h_img.hashFile(os.path.join(desc_dir, i))
                time_cal_list.append(time.time()-t0)
                encodingstring = h_img.getHexHashString()
                html_id = hashlib.md5('{0}{1}'.format(
                    time.time(), i).encode()).hexdigest()
                tmp_hash_file = '{0}.{1}.hashobj'.format(html_id, msg)
                hashobj_path = os.path.join(
                    app.config['UPLOADED_PATH'], tmp_hash_file)  # RETURN 哈希格式下载
                writePkl(hashobj_path, encoding)
                hashobj_path = '/download/'+tmp_hash_file
                hash_code_list[i] = (str(cnt), html_id, encoding, encodingstring,
                                     hashobj_path, os.path.join(icon_path_prefix, i))
                dict_filename_id[i] = str(cnt)
            file_avg_time = sum(time_cal_list)/len(time_cal_list)
            dict_simi_pair = {i: [] for i in hash_code_list}
            dict_simi_pair_info = {i: [] for i in hash_code_list}
            for i in hash_code_list:
                for j in hash_code_list:
                    if i != j:
                        tmp_h = hash_code_list[j]
                        tmp_simi = 1 - \
                            spatial.distance.cosine(
                                hash_code_list[i][2], tmp_h[2])
                        if tmp_simi > 0.85:
                            dict_simi_pair[i].append(tmp_h[0])
                            dict_simi_pair_info[i].append((j, tmp_simi))
            # 画图
            gexf_filename = session_filename[:-4]
            gexf = Gexf("Charles", "Network Structure")
            graph = gexf.addGraph("undirected", "static", "Network Sturcture")
            attr = graph.addNodeAttribute(
                'Modularity Class', '1', force_id='modularity_class')
            for i in hash_code_list:
                tmp_h = hash_code_list[i]
                node = graph.addNode(
                    tmp_h[0], i+'-'+tmp_h[0]+'-'+tmp_h[1]+'-'+tmp_h[5], r='255', g='255', b='255', size='30')
                node.addAttribute(attr, '0')
            graph_edge_set = set()
            for i in dict_simi_pair:
                for j in dict_simi_pair[i]:
                    tmp = (dict_filename_id[i], j)
                    if (tmp[1], tmp[0]) in graph_edge_set or tmp in graph_edge_set:
                        pass
                    else:
                        graph_edge_set.add(tmp)
            edgecnt = 0
            for i in graph_edge_set:
                graph.addEdge(edgecnt, i[0], i[1], weight='1')
                edgecnt += 1
            # 输出gexf图
            outfile = os.path.join(
                app.config['GRAPH_PATH'], '{0}.gexf'.format(gexf_filename))
            with open(outfile, 'wb') as f:
                gexf.write(f)
            return render_template('zip-img-result.html', gexf_filename=gexf_filename, file_count=file_count, file_avg_time=file_avg_time, hash_code_list=hash_code_list, dict_simi_pair_info=dict_simi_pair_info)
    else:
        flash(u'{0}功能尚未完善，敬请期待'.format(file_extensions))
        return redirect(url_for('index'))
    return 'okkkkkkkk'
@app.route('/test_result')
def test0():
    return render_template('result.html')
@app.route('/test_drawer')
def test1():
    return render_template('drawer.html')
@app.route('/test')
def test2():
    return render_template('txt-result.html')
@app.route('/test_collapse')
def test3():
    return render_template('collapse.html')
@app.route('/test_graph')
def test4():
    return render_template('graph-collapse.html')
@app.route('/test_jinja')
def test5():
    return render_template('jinja2.html', listss=[1, 2, 3], dicts={0: [1, 1, 1], '3': 4})
@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(directory=app.config['UPLOADED_PATH'], filename=filename)
@click.command()
@click.option('--debug', type=bool, default=True, help='define flask in debug mode or not')
@click.option('--port', type=int, default=8989, help='define flask running port')
def setting_stat(debug, port):
    """设置初始的状态
    1. debug: 设置flask是不是debug模式
    2. port: 设置flask开启服务的端口
    """
    app.run(debug=debug, host='0.0.0.0', port=port)
if __name__ == '__main__':
    setting_stat()
# -*- coding: utf-8 -*-
# Python version: 3.6
#     Gexf library in python
#     see gephi.org and gexf.net
#     repository : http://github.com/paulgirard/pygexf
#     documentation : http://packages.python.org/pygexf
#     main developper : Paul Girard, médialab Sciences Po
#     licence : GPL v3
from lxml import etree
from datetime import *
# missing features :
# load, validate, modify existing gexf file
# slices / phylogeny / ...
# evolution ideas :
# add display stats on graph composition when exportingto xml
# add anti-paralell edges test
# add a test based on existing example from gexf.net
# add modification accessors like setStart ...
# factorize attribute managment by creating an attribute class
# add a test code utility to check that code will not use _variable outside objects
class Gexf:
    def __init__(self, creator, description):
        self.creator = creator
        self.description = description
        self.graphs = []
        self.xmlns = "http://www.gephi.org/gexf/1.1draft"
        self.xsi = "http://www.w3.org/2001/XMLSchema-instance"
        self.schemaLocation = "http://www.gephi.org/gexf/1.1draft http://gephi.org/gexf/1.1draft.xsd"
        self.viz = "http://www.gexf.net/1.1draft/viz"
        self.version = "1.1"
    def addGraph(self, type, mode, label):
        g = Graph(type, mode, label)
        self.graphs.append(g)
        return g
    def getXML(self):
        gexfXML = etree.Element("{"+self.xmlns+"}gexf", version=self.version, nsmap={
                                None: self.xmlns, 'viz': self.viz, 'xsi': self.xsi})
# 		gexfXML.set("xmlnsxsi",)
        gexfXML.set("{xsi}schemaLocation", self.schemaLocation)
        meta = etree.SubElement(gexfXML, "meta")
        meta.set("lastmodified", datetime.now().isoformat())
        etree.SubElement(meta, "creator").text = self.creator
        etree.SubElement(meta, "description").text = self.description
        for graph in self.graphs:
            gexfXML.append(graph.getXML())
        return gexfXML
    def write(self, file):
        file.write(etree.tostring(self.getXML(),
                                  pretty_print=True, encoding='utf-8'))
        self.print_stat()
    def print_stat(self):
        for graph in self.graphs:
            graph.print_stat()
class Graph:
    def __init__(self, type, mode, label, start="", end=""):
        # control variable
        self.authorizedType = ("directed", "undirected")
        self.authorizedMode = ("dynamic", "static")
        self.defaultType = "directed"
        self.defaultMode = "static"
        self.label = label
        if type in self.authorizedType:
            self.type = type
        else:
            self.type = self.defaultType
        if mode in self.authorizedMode:
            self.mode = mode
        else:
            self.mode = self.defaultMode
        self.start = start
        self.end = end
        self._nodesAttributes = {}
        self._edgesAttributes = {}
        self._nodes = {}
        self._edges = {}
    def addNode(self, id, label, start="", end="", pid="", r="", g="", b="", size="", x="", y="", z=""):
        self._nodes[id] = Node(self, id, label, start,
                               end, pid, r, g, b, size, x, y, z)
        return self._nodes[id]
    def nodeExists(self, id):
        if id in list(self._nodes.keys()):
            return 1
        else:
            return 0
    def addEdge(self, id, source, target, weight="", start="", end="", label=""):
        self._edges[id] = Edge(self, id, source, target,
                               weight, start, end, label)
        return self._edges[id]
    def addNodeAttribute(self, title, defaultValue, type="integer", mode="static", force_id=""):
        # add to NodeAttributes
        # generate id
        if force_id == "":
            id = len(self._nodesAttributes)
        else:
            id = force_id
        self._nodesAttributes[id] = {"title": title, "default": defaultValue,
                                     "mode": mode, "type": type}		# modify Nodes with default
        #: bad idea and unecessary
        # for node in self._nodes.values():
        #	node.addAttribute(id,defaultValue)
        return id
    def addDefaultAttributesToNode(self, node):
        # add existing nodesattributes default values
        for id, values in self._nodesAttributes.items():
            node.addAttribute(id, values["default"])
    def checkNodeAttribute(self, id, value, start, end):
        # check conformity with type is missing
        if id in list(self._nodesAttributes.keys()):
            if self._nodesAttributes[id]["mode"] == "static" and (not start == "" or not end == ""):
                raise Exception(
                    "attribute "+str(id)+" is static you can't specify start or end dates. Declare Attribute as dynamic")
            return 1
        else:
            raise Exception(
                "attribute id unknown. Add Attribute to graph first")
    def addEdgeAttribute(self, title, defaultValue, type="integer", mode="static", force_id=""):
        # add to NodeAttributes
        # generate id
        if force_id == "":
            id = len(self._edgesAttributes)
        else:
            id = force_id
        self._edgesAttributes[id] = {"title": title, "default": defaultValue,
                                     "mode": mode, "type": type} 		# modify Nodes with default
        # for edge in self._edges.values():
        #	edge.addAttribute(id,defaultValue)
        return id
    def addDefaultAttributesToEdge(self, edge):
        # add existing nodesattributes default values
        for id, values in self._edgesAttributes.items():
            edge.addAttribute(id, values["default"])
    def checkEdgeAttribute(self, id, value, start, end):
        # check conformity with type is missing
        if id in list(self._edgesAttributes.keys()):
            if self._edgesAttributes[id]["mode"] == "static" and (not start == "" or not end == ""):
                raise Exception(
                    "attribute "+str(id)+" is static you can't specify start or end dates. Declare Attribute as dynamic")
            return 1
        else:
            raise Exception(
                "attribute id unknown. Add Attribute to graph first"
    def getXML(self):
        # return lxml etree element
        graphXML = etree.Element(
            "graph", defaultedgetype=self.type, mode=self.mode, label=self.label)
        attributesXMLNodeDynamic = etree.SubElement(graphXML, "attributes")
        attributesXMLNodeDynamic.set("class", "node")
        attributesXMLNodeDynamic.set("mode", "dynamic")
        attributesXMLNodeStatic = etree.SubElement(graphXML, "attributes")
        attributesXMLNodeStatic.set("class", "node")
        attributesXMLNodeStatic.set("mode", "static")
        for id, value in self._nodesAttributes.items():
            if value["mode"] == "static":
                attxml = attributesXMLNodeStatic
            else:
                attxml = attributesXMLNodeDynamic
            attributeXML = etree.SubElement(attxml, "attribute")
            attributeXML.set("id", str(id))
            attributeXML.set("title", value["title"])
            attributeXML.set("type", value["type"])
            etree.SubElement(attributeXML, "default").text = value["default"]
        attributesXMLEdgeDynamic = etree.SubElement(graphXML, "attributes")
        attributesXMLEdgeDynamic.set("class", "edge")
        attributesXMLEdgeDynamic.set("mode", "dynamic")
        attributesXMLEdgeStatic = etree.SubElement(graphXML, "attributes")
        attributesXMLEdgeStatic.set("class", "edge")
        attributesXMLEdgeStatic.set("mode", "static")
        for id, value in self._edgesAttributes.items():
            if value["mode"] == "static":
                attxml = attributesXMLEdgeStatic
            else:
                attxml = attributesXMLEdgeDynamic
            attributeXML = etree.SubElement(attxml, "attribute")
            attributeXML.set("id", str(id))
            attributeXML.set("title", value["title"])
            attributeXML.set("type", value["type"])
            etree.SubElement(attributeXML, "default").text = value["default"]
        nodesXML = etree.SubElement(graphXML, "nodes")
        for node in list(self._nodes.values()):
            nodesXML.append(node.getXML())
        edgesXML = etree.SubElement(graphXML, "edges")
        for edge in list(self._edges.values()):
            edgesXML.append(edge.getXML())
        return graphXML
    def print_stat(self):
        print(self.label+" "+self.type+" " +
              self.mode+" "+self.start+" "+self.end)
        print("number of nodes : "+str(len(self._nodes)))
        print("number of edges : "+str(len(self._edges)))
class Node:
    def __init__(self, graph, id, label, start="", end="", pid="", r="", g="", b="", size="", x="", y="", z=""):
        self.id = id
        self.label = label
        self.start = start
        self.end = end
        self.pid = pid
        self._graph = graph
        self.setColor(r, g, b)
        self.setPositionAndSize(size, x, y, z)
        if not self.pid == "":
            if not self._graph.nodeExists(self.pid):
                raise Exception("pid "+self.pid +
                                " node unknown, add nodes to graph first")
        self._attributes = []
        # add existing nodesattributes default values : bad idea and unecessary
        # self._graph.addDefaultAttributesToNode(self)
    def addAttribute(self, id, value, start="", end=""):
        if self._graph.checkNodeAttribute(id, value, start, end):
            self._attributes.append(
                {"id": id, "value": value, "start": start, "end": end})
    def getXML(self):
        # return lxml etree element
        try:
            nodeXML = etree.Element("node", id=str(
                self.id), label=str(self.label))
            if not self.start == "":
                nodeXML.set("start", self.start)
            if not self.end == "":
                nodeXML.set("end", self.end)
            if not self.pid == "":
                nodeXML.set("pid", self.pid)
            attributesXML = etree.SubElement(nodeXML, "attvalues")
            for atts in self._attributes:
                attributeXML = etree.SubElement(attributesXML, "attvalue")
                attributeXML.set("for", str(atts["id"]))
                attributeXML.set("value", atts["value"])
                if not atts["start"] == "":
                    attributeXML.set("start", atts["start"])
                if not atts["end"] == "":
                    attributeXML.set("end", atts["end"])
            if not self.size == "":
                # size : <viz:size value="66"/>
                colorXML = etree.SubElement(
                    nodeXML, "{http://www.gexf.net/1.1draft/viz}size")
                colorXML.set("value", self.size)
            if not self.x == "" and not self.y == "" and not self.z == "":
                # position : <viz:position x="239" y="173" z="66"/>
                colorXML = etree.SubElement(
                    nodeXML, "{http://www.gexf.net/1.1draft/viz}position")
                colorXML.set("x", self.x)
                colorXML.set("y", self.y)
                colorXML.set("z", self.z)
            if not self.r == "" and not self.g == "" and not self.b == "":
                # color : <viz:color r="239" g="173" b="66"/>
                colorXML = etree.SubElement(
                    nodeXML, "{http://www.gexf.net/1.1draft/viz}color")
                colorXML.set("r", self.r)
                colorXML.set("g", self.g)
                colorXML.set("b", self.b)
            return nodeXML
        except Exception as e:
            print(self.label)
            print(self._attributes)
            print(e)
            exit()
    def setColor(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
    def setPositionAndSize(self, size, x, y, z):
        self.size = size
        self.x = x
        self.y = y
        self.z = z
class Edge:
    def __init__(self, graph, id, source, target, weight="", start="", end="", label="", r="", g="", b=""):
        self.id = id
        self._graph = graph
        if self._graph.nodeExists(source):
            self._source = source
        else:
            raise Exception("source "+source +
                            " node unknown, add nodes to graph first")
        if self._graph.nodeExists(target):
            self._target = target
        else:
            raise Exception("target "+target +
                            " node unknown, add nodes to graph first")
        self.start = start
        self.end = end
        self.weight = weight
        self.label = label
        self._attributes = []
        # COLOR on edges isn't supported in GEXF
        self.setColor(r, g, b)
        # add existing nodesattributes default values : bad idea and unecessary
        # self._graph.addDefaultAttributesToEdge(self)
    def addAttribute(self, id, value, start="", end=""):
        if self._graph.checkEdgeAttribute(id, value, start, end):
            self._attributes.append(
                {"id": id, "value": value, "start": start, "end": end})
    def getXML(self):
        # return lxml etree element
        try:
            edgeXML = etree.Element("edge", id=str(self.id), source=str(
                self._source), target=str(self._target))
            if not self.start == "":
                edgeXML.set("start", self.start)
            if not self.end == "":
                edgeXML.set("end", self.end)
            if not self.weight == "":
                edgeXML.set("weight", str(self.weight))
            if not self.label == "":
                edgeXML.set("label", str(self.label))
# COLOR on edges isn't supported in GEXF
            if not self.r == "" and not self.g == "" and not self.b == "":
                # color : <viz:color r="239" g="173" b="66"/>
                colorXML = etree.SubElement(
                    edgeXML, "{http://www.gexf.net/1.1draft/viz}color")
                colorXML.set("r", self.r)
                colorXML.set("g", self.g)
                colorXML.set("b", self.b)
            attributesXML = etree.SubElement(edgeXML, "attvalues")
            for atts in self._attributes:
                attributeXML = etree.SubElement(attributesXML, "attvalue")
                attributeXML.set("for", str(atts["id"]))
                attributeXML.set("value", atts["value"])
                if not atts["start"] == "":
                    attributeXML.set("start", atts["start"])
                if not atts["end"] == "":
                    attributeXML.set("end", atts["end"])
            return edgeXML
        except Exception as e:
            print(self._source+" "+self._target)
            print(e)
            exit()
# COLOR on edges isn't supported in GEXF
    def setColor(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
class GexfImport:
    # class coded by elie Rotenberg, médialab 20/07/2010
    def __init__(self, file_like):
        parser = etree.XMLParser(ns_clean=True)
        tree = etree.parse(file_like, parser)
        gexf_xml = tree.getroot()
        tag = self.ns_clean(gexf_xml.tag).lower()
        if tag != "gexf":
            self.msg_unexpected_tag("gexf", tag)
            return
        self.gexf_obj = None
        for child in gexf_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "meta":
                meta_xml = child
                self.gexf_obj = self.extract_gexf_obj(meta_xml)
            if tag == "graph":
                graph_xml = child
                if self.gexf_obj == None:
                    self.msg_unexpected_tag("meta", tag)
                    return
                self.graph_obj = self.extract_graph_obj(graph_xml)
    def ns_clean(self, token):
        i = token.find('}')
        return token[i+1:]
    def msg_unexpected_tag(self, expected, got):
        print("Error : incorrect xml. Expected tag {expected}, not {got}.".format(
            expected=expected, got=got))
    def extract_gexf_obj(self, meta_xml):
        for child in meta_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "creator":
                creator = child.text
            if tag == "description":
                description = child.text
        return Gexf(creator=creator, description=description)
    def extract_graph_obj(self, graph_xml):
        type = ""
        mode = ""
        label = ""
        for attr in graph_xml.attrib:
            attr = attr.lower()
            if attr == "defaultedgetype":
                type = graph_xml.attrib[attr]
            if attr == "mode":
                mode = graph_xml.attrib[attr]
            if attr == "label":
                label = graph_xml.attrib[attr]
        self.graph_obj = self.gexf_obj.addGraph(
            type=type, mode=mode, label=label)
        for child in graph_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "attributes":
                attributes_xml = child
                self.extract_attributes(attributes_xml)
            if tag == "nodes":
                nodes_xml = child
                self.extract_nodes(nodes_xml)
            if tag == "edges":
                edges_xml = child
                self.extract_edges(edges_xml)
    def extract_attributes(self, attributes_xml):
        attr_class = None
        mode = ""
        for attr in attributes_xml.attrib:
            attr = attr.lower()
            if attr == "class":
                attr_class = attributes_xml.attrib[attr].lower()
            if attr == "mode":
                mode = attributes_xml.attrib[attr]
        for child in attributes_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "attribute":
                attribute_xml = child
                self.extract_attribute(attribute_xml, attr_class, mode)
    def extract_attribute(self, attribute_xml, attr_class, mode):
        id = ""
        title = ""
        type = ""
        for attr in attribute_xml.attrib:
            attr = attr.lower()
            if attr == "id":
                id = attribute_xml.attrib[attr]
            if attr == "title":
                title = attribute_xml.attrib[attr]
            if attr == "type":
                type = attribute_xml.attrib[attr]
        default = ""
        for child in attribute_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "default":
                default = child.text
        if attr_class == "node":
            self.graph_obj.addNodeAttribute(
                title, default, type, mode, force_id=id)
        if attr_class == "edge":
            self.graph_obj.addEdgeAttribute(
                title, default, type, mode, force_id=id)
    def extract_nodes(self, nodes_xml):
        for child in nodes_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "node":
                node_xml = child
                self.extract_node(node_xml)
    def extract_node(self, node_xml):
        id = ""
        label = ""
        start = ""
        end = ""
        pid = ""
        r = ""
        g = ""
        b = ""
        for attr in node_xml.attrib:
            attr = attr.lower()
            if attr == "id":
                id = node_xml.attrib[attr]
            if attr == "label":
                label = node_xml.attrib[attr]
            if attr == "start":
                start = node_xml.attrib[attr]
            if attr == "end":
                start = node_xml.attrib[attr]
            if attr == "pid":
                pid = node_xml.attrib[attr]
        attvalues_xmls = []
        for child in node_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "attvalues":
                attvalues_xmls.append(child)
            if tag == "viz:color":
                r = child.attrib["r"]
                g = child.attrib["g"]
                b = child.attrib["b"]
        self.node_obj = self.graph_obj.addNode(
            id=id, label=label, start=start, end=end, pid=pid, r=r, g=g, b=b)
        for attvalues_xml in attvalues_xmls:
            self.extract_node_attvalues(attvalues_xml)
    def extract_node_attvalues(self, attvalues_xml):
        for child in attvalues_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "attvalue":
                attvalue_xml = child
                self.extract_node_attvalue(attvalue_xml)
    def extract_node_attvalue(self, attvalue_xml):
        id = ""
        value = ""
        start = ""
        end = ""
        for attr in attvalue_xml.attrib:
            attr = attr.lower()
            if attr == "for":
                id = attvalue_xml.attrib[attr]
            if attr == "value":
                value = attvalue_xml.attrib[attr]
            if attr == "start":
                start = attvalue_xml.attrib[attr]
            if attr == "end":
                end = attvalue_xml.attrib[attr]
        self.node_obj.addAttribute(id=id, value=value, start=start, end=end)
    def extract_edges(self, edges_xml):
        for child in edges_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "edge":
                edge_xml = child
                self.extract_edge(edge_xml)
    def extract_edge(self, edge_xml):
        id = ""
        source = ""
        target = ""
        weight = ""
        start = ""
        end = ""
        label = ""
        for attr in edge_xml.attrib:
            attr = attr.lower()
            if attr == "id":
                id = edge_xml.attrib[attr]
            if attr == "source":
                source = edge_xml.attrib[attr]
            if attr == "target":
                target = edge_xml.attrib[attr]
            if attr == "weight":
                weight = edge_xml.attrib[attr]
            if attr == "start":
                start = edge_xml.attrib[attr]
            if attr == "end":
                end = edge_xml.attrib[attr]
            if attr == "label":
                label = edge_xml.attrib[attr]
        self.edge_obj = self.graph_obj.addEdge(
            id=id, source=source, target=target, weight=weight, start=start, end=end, label=label)
        for child in edge_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "attvalues":
                attvalues_xml = child
                self.extract_edge_attvalues(attvalues_xml)
    def extract_edge_attvalues(self, attvalues_xml):
        for child in attvalues_xml:
            tag = self.ns_clean(child.tag).lower()
            if tag == "attvalue":
                attvalue_xml = child
                self.extract_edge_attvalue(attvalue_xml)
#	def addAttribute(self,id,value,start="",end="") :
    def extract_edge_attvalue(self, attvalue_xml):
        id = ""
        value = ""
        start = ""
        end = ""
        for attr in attvalue_xml.attrib:
            if attr == "for":
                id = attvalue_xml.attrib[attr]
            if attr == "value":
                value = attvalue_xml.attrib[attr]
            if attr == "start":
                start = attvalue_xml.attrib[attr]
            if attr == "end":
                end = attvalue_xml.attrib[attr]
        self.edge_obj.addAttribute(id=id, value=value, start=start, end=end)
    def gexf(self):
        return self.gexf_obj
# coding=utf-8
# Python version: 3.6
import os
import logging
import ffmpeg
import cv2
import random
from datasketch import MinHash
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from PIL import Image
from colorama import Fore, Style
# 设置log
logging.basicConfig(
    level=logging.INFO,
    format=''.join([Style.DIM, '[%(asctime)s] ',
                    Style.NORMAL, '%(message)s',
                    Style.RESET_ALL]),
    datefmt='%m-%d %T',
)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
def cropImage(src, des, filename=None):
    """
    利用PIL随机剪切图片长宽的 0% - 25%
    :param src: source picture file path
    :param des: destination picture store directory path
    :param filename: destination picture file name
    :return: bool, True for success
    """
    FIXED_RATIO = 0.25
    if os.path.isfile(src):
        im = Image.open(src)
        box = []
        for i in im.size:
            l = random.randint(0, int(i * FIXED_RATIO))
            a = random.randint(0, l)
            b = i - l + a
            box += [a, b]
        box[1], box[2] = box[2], box[1]
        crop = im.crop(box)
        if os.path.isdir(des):
            pass
        else:
            logger.warning(Fore.YELLOW + des + ': no such directory.')
            os.mkdir(des)
            logger.info(Fore.BLUE + 'os mkdir ' + des)
        if filename is None:
            filename = 'default_crop.jpeg'
            logger.warning(
                Fore.YELLOW + 'saved file has been set to default_crop.jpeg')
        try:
            crop.save(os.path.join(des, filename))
            logger.info(
                Fore.BLUE + 'cropped image range (left, upper, right, lower) {0}'.format(box))
        except Exception as exc:
            logger.error(
                Fore.RED + 'save {0} failed with error {1}'.format(os.path.join(des, filename), exc))
            return False
        return True
    else:
        logger.error(Fore.RED + src + ': no such file.')
        return False
def rotateImage(src, des, filename=None):
    """
    利用PIL随机旋转图片长宽的 0-25 度
    :param src: source picture file path
    :param des: destination picture store directory path
    :param filename: destination picture file name
    :return: None
    """
    FIXED_ROTATION = 25
    if os.path.isfile(src):
        im = Image.open(src)
        _degree = random.randint(0, FIXED_ROTATION)
        rotate = im.rotate(_degree)
        if os.path.isdir(des):
            pass
        else:
            logger.warning(Fore.YELLOW + des + ': no such directory.')
            os.mkdir(des)
            logger.info(Fore.BLUE + 'os mkdir ' + des)
        if filename is None:
            filename = 'default_rotate.jpeg'
            logger.warning(
                Fore.YELLOW + 'saved file has been set to default_rotate.jpeg')
        try:
            rotate.save(os.path.join(des, filename))
            logger.info(Fore.BLUE + 'rotated image {0} degree'.format(_degree))
        except Exception as exc:
            logger.error(
                Fore.RED + 'save {0} failed with error {1}'.format(os.path.join(des, filename), exc))
            return False
        return True
    else:
        logger.error(Fore.RED + src + ': no such file.')
        return False
def mergeImage(src, des, filename=None):
    """
    利用PIL横向合并图片，对于不同大小的图片，先将大图片压缩至最小图片的尺度，再合并
    :param src: list, source picture file path list
    :param des: destination picture store directory path
    :param filename: destination picture file name
    :return: None
    """
    for i in src:
        if not os.path.isfile(i):
            logger.error(Fore.RED + i + ': no such file.')
            return False
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    imgs = [Image.open(i) for i in src]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    imgs_comb = Image.fromarray(imgs_comb)
    if os.path.isdir(des):
        pass
    else:
        logger.warning(Fore.YELLOW + des + ': no such directory.')
        os.mkdir(des)
        logger.info(Fore.BLUE + 'os mkdir ' + des)
    if filename is None:
        filename = 'default_merge.jpeg'
        logger.warning(
            Fore.YELLOW + 'saved file has been set to default_merge.jpeg')
    try:
        imgs_comb.save(os.path.join(des, filename))
        logger.info(Fore.BLUE + 'merged image {0} size'.format(min_shape))
    except Exception as exc:
        logger.error(
            Fore.RED + 'save {0} failed with error {1}'.format(os.path.join(des, filename), exc))
        return False
    return True
def SIFT(src):
    """
    利用cv2图片SIFT获得特征
    :param src: source picture file path
    :return: keypoint and descriptor
    """
    if os.path.isfile(src):
        img = cv2.imread(os.path.abspath(src), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des
    else:
        logger.error(Fore.RED + src + ': no such file.')
def getVideoInfo(src):
    """
    利用FFmpeg获取视频文件信息
    :param src: source video file path
    :return: dict ffmpeg video stream info
    """
    if os.path.isfile(src):
        probe = ffmpeg.probe(src)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream
    else:
        logger.error(Fore.RED + src + ': no such file.')
def keyFrameExtraction(src, des):
    """
    利用FFmpeg进行关键帧提取
    :param src: source video file path
    :param des: destination keyframe store directory path
    :return: bool, True for success
    """
    if os.path.isfile(src):
        _input = ffmpeg.input(src)
        _filter = _input.filter('select', 'eq(pict_type,PICT_TYPE_I)')
        if os.path.isdir(des):
            pass
        else:
            logger.warning(Fore.YELLOW + des + ': no such directory.')
            os.mkdir(des)
            logger.info(Fore.BLUE + 'os mkdir ' + des)
        _output = _filter.output(os.path.join(
            des, 'keyframe_%02d.jpeg'), format='image2', vsync=2)
        _out, _err = _output.run()
        if _err:
            logger.error(Fore.RED + 'ffmpeg error: ' + _err)
            return False
        return True
    else:
        logger.error(Fore.RED + src + ': no such file.')
        return False
def cutVideo(src, des, filename=None, start_timestamp=0, duration=10):
    """
    利用FFmpeg进行视频切割
    :param src: source video file path
    :param des: destination keyframe store directory path
    :param filename: output file name
    :param start_timestamp: start to crop video
    :param duration: output video length, should more than 10 seconds
    :return: bool, True for success
    """
    if os.path.isfile(src):
        _input = ffmpeg.input(src)
        if os.path.isdir(des):
            pass
        else:
            logger.warning(Fore.YELLOW + des + ': no such directory.')
            os.mkdir(des)
            logger.info(Fore.BLUE + 'os mkdir ' + des)
        if filename is None:
            filename = 'default_crop.mp4'
            logger.warning(
                Fore.YELLOW + 'saved file has been set to default_crop.mp4')
        _output = _input.output(os.path.join(
            des, filename), ss=start_timestamp, t=duration, c='copy')
        _out, _err = _output.run()
        if _err:
            logger.error(Fore.RED + 'ffmpeg error: ' + _err)
            return False
        return True
    else:
        logger.error(Fore.RED + src + ': no such file.')
        return False
def mergeVideo(src1, src2, des, filename=None):
    """
    利用FFmpeg进行两个视频的合并
    :param src1: source video file path
    :param src2: source video file path
    :param des: destination keyframe store directory path
    :param filename: output file name
    :return: bool, True for success
    """
    if os.path.isfile(src1) and os.path.isfile(src2):
        _input1 = ffmpeg.input(src1)
        _input2 = ffmpeg.input(src2)
        if os.path.isdir(des):
            pass
        else:
            logger.warning(Fore.YELLOW + des + ': no such directory.')
            os.mkdir(des)
            logger.info(Fore.BLUE + 'os mkdir ' + des)
        if filename is None:
            filename = 'default_merge.mp4'
            logger.warning(
                Fore.YELLOW + 'saved file has been set to default_merge.mp4')
        _joined = ffmpeg.concat(
            _input1['v'], _input1['a'], _input2['v'], _input2['a'], v=1, a=1)
        _output = _joined.output(os.path.join(des, filename))
        _out, _err = _output.run()
        if _err:
            logger.error(Fore.RED + 'ffmpeg error: ' + _err)
            return False
        return True
    else:
        logger.error(Fore.RED + src1 + 'or' + 'src2' + ': no such file.')
        return False
def getImageHashValues(sift_des):
    """
    利用datasketch得到图片的Hash值
    :param sift_des: OpenCV SIFT 后得到的des值，它应该是[n, 128]的向量
    :return: numpy.ndarray datasketch 计算得到的Hash值
    """
    assert type(
        sift_des) == np.ndarray, 'INPUT sift_des should be numpy.ndarray!'
    m = MinHash(num_perm=1024)
    for i in sift_des:
        assert type(
            i) == np.ndarray, 'INPUT sift_des[i] should be numpy.ndarray!'
        m.update(i.tostring())
    return m.hashvalues
def getImageJaccardMeasure(sift_des1, sift_des2):
    return None
def jaccardMeasure(hashvalue1, hashvalue2):
    """
    利用datasketch计算两个Hash值的Jaccard相似度
    :param hashvalue1: datasketch 哈希值1
    :param hashvalue2: datasketch 哈希值2
    :return: float 两个哈希值的相似度 [0, 1]
    """
    try:
        m1 = MinHash(hashvalues=hashvalue1)
        m2 = MinHash(hashvalues=hashvalue2)
        return m1.jaccard(m2)
    except Exception as exc:
        logger.error(Fore.RED + 'MinHash failed with {0}'.format(exc))
        return 0.0
def getTextHashValues(text, ngram=5):
    """
    利用datasketch和NLTK得到文本的Hash值
    :param text: 一段字符串文本
    :param ngram: n-gram的值，短文本推荐为5
    :return: numpy.ndarray datasketch 计算得到的Hash值
    """
    assert type(text) == str, 'INPUT text should be str'
    m = MinHash()
    for i in nltk.ngrams(text, ngram):
        m.update(''.join(i).encode('utf8'))
    return m.hashvalues
def removeSentences(text, num=2):
    """
    利用NLTK分句，随机删除一部分句子
    :param text: str, 输入文本
    :param num: int, 删除句子的数量
    :return: str 处理后的文本
    """
    assert type(text) == str, 'INPUT text should be str'
    sen = nltk.sent_tokenize(text)
    l = len(sen)
    assert l > num, 'INPUT sentences are too less to remove'
    mask = np.ones(l)
    mask[:num] = 0
    mask = np.random.permutation(mask)
    logger.info(Fore.BLUE + 'Delete {0} sentence(s): '.format(
        l - sum(mask)) + ' $-$ '.join([sen[i] for i in range(l) if mask[i] == 0]))
    return ' '.join([sen[i] for i in range(l) if mask[i] == 1])
def removeWords(text, fraction=0.1):
    """
    利用NLTK分词，随机删除一部分单词
    :param text: str, 输入文本
    :param fraction: float, 删除单词的数量比例
    :return: str 处理后的文本
    """
    assert type(text) == str, 'INPUT text should be str'
    words = nltk.word_tokenize(text)
    l = len(words)
    mask = np.ones(l)
    mask[:int(l * fraction)] = 0
    mask = np.random.permutation(mask)
    logger.info(Fore.BLUE + 'Delete {0} word(s): '.format(
        l - sum(mask)) + ' , '.join([words[i] for i in range(l) if mask[i] == 0]))
    return ' '.join([words[i] for i in range(l) if mask[i] == 1])
def penn_to_wn(tag):
    """将tag词性和语义转换一下"""
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
def pos_tag_with_synset(tagged):
    """将词性标签的tag转换为语义标签的tag
    :param tagged: nltk.pos_tag 的返回结果
    :return: tuple list, 语义标签
    """
    synsets = []
    lemmatzr = nltk.stem.WordNetLemmatizer()
    for token in tagged:
        wn_tag = penn_to_wn(token[1])
        if not wn_tag:
            synsets.append((token[0], None))  # 没有对应的语义
            continue
        lemma = lemmatzr.lemmatize(token[0], pos=wn_tag)
        syn = wn.synsets(lemma, pos=wn_tag)
        if(len(syn) > 0):
            synsets.append((token[0], syn[0]))  # 选择词性约束下的第一个语义
        else:
            synn = wn.synsets(lemma)
            if(len(synn) > 0):
                synsets.append((token[0], synn[0]))
            else:
                synsets.append((token[0], None))
    return synsets
def switchAntonyms(text, fraction=0.1):
    """
    利用NLTK分词，随机替换一部分词为反义词
    :param text: str, 输入文本
    :param fraction: float, 替换单词的数量比例，由于某些词没有反义词所以实际数量会比设定的要少
    :return: str 处理后的文本
    """
    assert type(text) == str, 'INPUT text should be str'
    words = nltk.word_tokenize(text)
    l = len(words)
    mask = np.ones(l)
    mask[:int(l * fraction)] = 0
    mask = np.random.permutation(mask)
    synsets = pos_tag_with_synset(nltk.pos_tag(words))
    antsets = []
    count = 0
    for i in range(l):
        if(mask[i] == 0):
            if(synsets[i][1] is not None):
                syn = synsets[i][1]
                if(syn.lemmas()):
                    flag = False
                    for j in syn.lemmas():
                        if j.antonyms():
                            # 选取排名第一个的反义词
                            antsets.append(j.antonyms()[0].name())
                            count += 1
                            flag = True
                            break
                    if(not flag):
                        antsets.append(synsets[i][0])
                else:
                    antsets.append(synsets[i][0])
            else:
                antsets.append(synsets[i][0])
        else:
            antsets.append(synsets[i][0])
    logger.info(Fore.BLUE + 'Switch {0} word(s): '.format(count))
return ' '.join(antsets)
#encoding:utf-8
# Python version: 3.6
import time
import os
import numpy as np
import random
from libKMCUDA import kmeans_cuda
n_clusters = 30000
iterations = 60
p = './batch_40/'
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
def minibatch_kmeans(center, data, n_clusters):
    """
    分batch的kmenas
    data: 一个batch的data
    """
    t0 = time.time()
    centroids,_=kmeans_cuda(data, n_clusters, init=center, verbosity=1,yinyang_t=0)
    t_mini_batch = time.time() - t0
    print('This iteration takes ', t_mini_batch, 's')
    return centroids
def firstbatch_kmeans(data, n_clusters):
    """
    分batch的kmenas
    data: 一个batch的data
    """
    t0 = time.time()
    centroids,_=kmeans_cuda(data, n_clusters, verbosity=1, device=0,yinyang_t=0)
    t_mini_batch = time.time() - t0
    print('This iteration takes ', t_mini_batch, 's')
    return centroids
def load_data(iteration):
    """
    根据目前的迭代轮数导入数据
    先把所有数据过一遍
    然后随机选择两个文件按50%的概率抽样组合成新的数据返回
    """
    src = [os.path.join(p,i) for i in sorted(listdir_nohidden(p))]
    if iteration<len(src):
        print('loading: ', os.path.join(p, str(iteration)+'.npy'))
        rdata = np.load(os.path.join(p, str(iteration)+'.npy'))
        return rdata
    else:
        rdata = []
        select = random.sample(src,2)
        print('loading: ', select)
        for s in select:
            data = np.load(s)
            for i in data:
                if random.random()<0.5:
                    rdata.append(i)
            del data
        rdata=np.array(rdata)
        return rdata
if __name__ =='__main__':
    np.random.seed(0)
    print('中文测试')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    # center = np.load('centroid/0-iteration.npy')
    center = firstbatch_kmeans(load_data(0), n_clusters)
    np.save('centroid/0-iteration.npy',center)
    for i in range(1,iterations):
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        print('iteration:',i)
        data = load_data(i)
        center = minibatch_kmeans(center, data, n_clusters)
        np.save('centroid/'+str(i)+'-iteration.npy',center)
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import time
import numpy as np
from scipy.spatial import distance
from eucl_dist.gpu_dist import dist as gdist
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    #  d = distance.cdist(baseline, vec, metric)
    d = gdist(baseline, vec, optimize_level=3)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
flickr = './sift/'
num = [str(i) for i in range(100)]
if __name__ == '__main__':
    counter = 0
    npy = None
    name = 0
    for i in num:
        first_dir = os.path.join(flickr, i)
        print('file dir', i)
        for c, j in enumerate(sorted(os.listdir(first_dir))):
            if c % 1000 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), ' file dir:', i, c, flush=True)
            tmp = np.load(os.path.join(first_dir, j))
            nn = nearlist_vector(tmp)
            np.save(os.path.join(os.path.join('./image_nn', i), j), nn)
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import time
import numpy as np
from scipy.spatial import distance
from eucl_dist.gpu_dist import dist as gdist
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    #  d = distance.cdist(baseline, vec, metric)
    d = gdist(baseline, vec, optimize_level=3)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
flickr = './sift/'
file_dir = input()
num = [file_dir]
if __name__ == '__main__':
    counter = 0
    npy = None
    name = 0
    for i in num:
        first_dir = os.path.join(flickr, i)
        print('file dir', i)
        for c, j in enumerate(sorted(os.listdir(first_dir))):
            if c % 1000 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S"),
                      ' file dir:', i, c, flush=True)
            tmp = np.load(os.path.join(first_dir, j))
            nn = nearlist_vector(tmp)
            np.save(os.path.join(os.path.join('./image_nn', i), j), nn)
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import sqlite3
from utils import getImageHashValues
import numpy as np
class db():
    def __init__(self, db='datasets.db'):
        self.db = db
    def connect(self):
        self.conn = sqlite3.connect(self.db)
    def close(self):
        self.conn.commit()
        self.conn.close()
    def init_flickr(self):
        self.connect()
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IMDS (ID INTEGER PRIMARY KEY AUTOINCREMENT, NAME CHAR(20), ADDRESS char(80) NOT NULL, HASH BLOB NOT NULL);''')
        image_root = 'Image/mirflickr1m/images/'
        vector_root = 'image_nn'
        for d in os.listdir(vector_root)[0:2]:
            _path = os.path.join(vector_root, d)
            print(_path)
            for f in os.listdir(_path):
                _file = os.path.join(_path, f)
                if os.path.isfile(_file):
                    if _file[-4:] == '.npy':
                        index = _file.rfind('/')
                        filename = _file[index+1:-4]
                        filedir = str(int(_file[index+1:-8])//10000)+'/'
                        data = np.load(_file)
                        c.execute("insert into IMDS values (null, ?, ?, ?)", (filename, image_root+filedir+filename, getImageHashValues(data).tobytes()))
        self.close()
    def delete_all(self, table_name):
        self.connect()
        c = self.conn.cursor()
        c.execute('''delete from '''+table_name)
        self.close()
    def delete_table(self, table_name):
        self.connect()
        c = self.conn.cursor()
        c.execute('''drop table '''+table_name)
        self.close()
if __name__ == '__main__':
    d = db()
    d.delete_table('IMDS')
d.init_flickr()
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from utils import cropImage, jaccardMeasure, getImageHashValues
from eucl_dist.gpu_dist import dist as gdist
from datasketch import MinHashLSHForest, MinHash
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    d = gdist(baseline, vec, optimize_level=3)
    #  d = distance.cdist(baseline, vec, metric)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
def SIFT(src):
    """
    利用cv2图片SIFT获得特征
    :param src: source picture file path
    :return: keypoint and descriptor
    """
    if os.path.isfile(src):
        img = cv2.imread(os.path.abspath(src), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des
    else:
        print(src + ': no such file.')
with gzip.open('hashes_flickr.pkl.gz', 'rb') as f:
    image_hash = pickle.load(f)
with gzip.open('filckr_forest_lean.pkl.gz', 'rb') as f:
    image_hash_index = pickle.load(f)
if __name__ == '__main__':
    crops = []
    for i in paths[7002:]:
        for j in range(2):
            filedir = 'crop'
            filename = str(j)+'-'+i[i.rfind('/')+1:]
            cropImage(i, filedir, filename)
            _, sift = SIFT(os.path.join(filedir, filename))
            if sift is None:
                continue
            nn = nearlist_vector(sift)
            h = getImageHashValues(nn)
            try:
                t = image_hash_index.query(MinHash(hashvalues=h), 1)[0]
            except:
                continue
                os.remove(os.path.join(filedir, filename))
            s = jaccardMeasure(h, image_hash[t])
            print(t, filename, s)
            if t in filename:
                crops.append([os.path.join(filedir, filename), s])
                with gzip.open('crops.info.pkl.gz', 'wb') as f:
                    pickle.dump(obj=crops, file=f)
            else:
                os.remove(os.path.join(filedir, filename))
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from utils import rotateImage, jaccardMeasure, getImageHashValues
from eucl_dist.gpu_dist import dist as gdist
from datasketch import MinHashLSHForest, MinHash
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    d = gdist(baseline, vec, optimize_level=3)
    #  d = distance.cdist(baseline, vec, metric)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
def SIFT(src):
    """
    利用cv2图片SIFT获得特征
    :param src: source picture file path
    :return: keypoint and descriptor
    """
    if os.path.isfile(src):
        img = cv2.imread(os.path.abspath(src), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des
    else:
        print(src + ': no such file.')
with gzip.open('hashes_flickr.pkl.gz', 'rb') as f:
    image_hash = pickle.load(f)
with gzip.open('filckr_forest_lean.pkl.gz', 'rb') as f:
    image_hash_index = pickle.load(f)
if __name__ == '__main__':
    rotates = []
    for i in paths[20000:]:
        for j in range(2):
            filedir = 'rotate'
            filename = str(j)+'-'+i[i.rfind('/')+1:]
            try:
                rotateImage(i, filedir, filename)
                _, sift = SIFT(os.path.join(filedir, filename))
                if sift is None:
                    continue
                nn = nearlist_vector(sift)
                h = getImageHashValues(nn)
                t = image_hash_index.query(MinHash(hashvalues=h), 1)[0]
            except:
                os.remove(os.path.join(filedir, filename))
                continue
            s = jaccardMeasure(h, image_hash[t])
            print(t, filename, s)
            if t in filename:
                rotates.append([os.path.join(filedir, filename), s])
                with gzip.open('rotates.info.pkl.gz', 'wb') as f:
                    pickle.dump(obj=rotates, file=f)
            else:
                os.remove(os.path.join(filedir, filename))
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from utils import cropImage, jaccardMeasure, getImageHashValues
from eucl_dist.gpu_dist import dist as gdist
from datasketch import MinHashLSHForest, MinHash
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    d = gdist(baseline, vec, optimize_level=3)
    #  d = distance.cdist(baseline, vec, metric)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
def SIFT(src):
    """
    利用cv2图片SIFT获得特征
    :param src: source picture file path
    :return: keypoint and descriptor
    """
    if os.path.isfile(src):
        img = cv2.imread(os.path.abspath(src), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des
    else:
        print(src + ': no such file.')
with gzip.open('hashes_flickr.pkl.gz', 'rb') as f:
    image_hash = pickle.load(f)
if __name__ == '__main__':
    crops = []
    for cnt,i in enumerate(paths):
        if cnt%1000==0:
            np.save('benchmark_crops.npy',crops)
        for j in range(4):
            filedir = 'benchmark_crop'
            filename = i[i.rfind('/')+1:]
            try:
                cropImage(i, filedir, filename)
                _, sift = SIFT(os.path.join(filedir, filename))
                if sift is None:
                    continue
                nn = nearlist_vector(sift)
                h = getImageHashValues(nn)
                os.remove(os.path.join(filedir, filename))
                s = jaccardMeasure(h, image_hash[filename])
            except Exception as e:
                continue
            crops.append(s)
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from utils import rotateImage, jaccardMeasure, getImageHashValues
from eucl_dist.gpu_dist import dist as gdist
from datasketch import MinHashLSHForest, MinHash
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    d = gdist(baseline, vec, optimize_level=3)
    #  d = distance.cdist(baseline, vec, metric)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
def SIFT(src):
    """
    利用cv2图片SIFT获得特征
    :param src: source picture file path
    :return: keypoint and descriptor
    """
    if os.path.isfile(src):
        img = cv2.imread(os.path.abspath(src), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des
    else:
        print(src + ': no such file.')
with gzip.open('hashes_flickr.pkl.gz', 'rb') as f:
    image_hash = pickle.load(f)
if __name__ == '__main__':
    rotates = []
    for cnt,i in enumerate(paths):
        if cnt%1000==0:
            np.save('benchmark_rotates.npy',rotates)
        for j in range(4):
            filedir = 'benchmark_rotate'
            filename = i[i.rfind('/')+1:]
            try:
                rotateImage(i, filedir, filename)
                _, sift = SIFT(os.path.join(filedir, filename))
                if sift is None:
                    continue
                nn = nearlist_vector(sift)
                h = getImageHashValues(nn)
                s = jaccardMeasure(h, image_hash[filename])
                os.remove(os.path.join(filedir, filename))
            except Exception as e:
                continue
            rotates.append(s)
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from utils import rotateImage, jaccardMeasure, getImageHashValues
from eucl_dist.gpu_dist import dist as gdist
from datasketch import MinHashLSHForest, MinHash
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    d = gdist(baseline, vec, optimize_level=3)
    #  d = distance.cdist(baseline, vec, metric)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
def SIFT(src):
    """
    利用cv2图片SIFT获得特征
    :param src: source picture file path
    :return: keypoint and descriptor
    """
    if os.path.isfile(src):
        img = cv2.imread(os.path.abspath(src), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des
    else:
        print(src + ': no such file.')
if __name__ == '__main__':
    intra = []
    for index, i in enumerate(paths):
        for cnt, j in enumerate(paths[index+1:]):
            if cnt % 1000 == 0:
                np.save('flickr_intra.npy', intra)
                print(cnt)
            for _ in range(4):
                try:
                    _, sift = SIFT(i)
                    if sift is None:
                        continue
                    nn = nearlist_vector(sift)
                    h1 = getImageHashValues(nn)
                    _, sift = SIFT(j)
                    if sift is None:
                        continue
                    nn = nearlist_vector(sift)
                    h2 = getImageHashValues(nn)
                    s = jaccardMeasure(h1, h2)
                except Exception as e:
                    print(e)
                    continue
                intra.append(s)
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from utils import cropImage
from imagededup.methods import PHash, AHash, CNN, DHash, WHash
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
if __name__ == '__main__':
    cnn = CNN()
    crops = [[], [], [], [], []]
    for cnt,i in enumerate(paths):
        if cnt%1000==0:
            np.save('imagededup_crop.npy',crops)
        for j in range(4):
            filedir = 'benchmark_crop'
            filename = i[i.rfind('/')+1:]
            try:
                cropImage(i, filedir, filename)
                # AHash
                phasher = AHash()
                encoding1 = phasher.encode_image(image_file=i)
                encoding2 = phasher.encode_image(image_file=os.path.join(filedir, filename))
                dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                if dup['1']:
                    crops[0].append(dup['1'][0][1])
                else:
                    crops[0].append(64)
                # PHash
                phasher = PHash()
                encoding1 = phasher.encode_image(image_file=i)
                encoding2 = phasher.encode_image(image_file=os.path.join(filedir, filename))
                dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                if dup['1']:
                    crops[1].append(dup['1'][0][1])
                else:
                    crops[1].append(64)
                # DHash
                phasher = DHash()
                encoding1 = phasher.encode_image(image_file=i)
                encoding2 = phasher.encode_image(image_file=os.path.join(filedir, filename))
                dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                if dup['1']:
                    crops[2].append(dup['1'][0][1])
                else:
                    crops[2].append(64)
                # WHash
                phasher = WHash()
                encoding1 = phasher.encode_image(image_file=i)
                encoding2 = phasher.encode_image(image_file=os.path.join(filedir, filename))
                dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                if dup['1']:
                    crops[3].append(dup['1'][0][1])
                else:
                    crops[3].append(64)
                # CNN
                encoding1 = cnn.encode_image(image_file=i)
                encoding2 = cnn.encode_image(image_file=os.path.join(filedir, filename))
                dup = cnn.find_duplicates(encoding_map={'1':encoding1[0],'2':encoding2[0]}, min_similarity_threshold=-1.0, scores=True)
                if dup['1']:
                    crops[4].append(dup['1'][0][1])
                else:
                    crops[4].append(-1.0)
                del encoding1, encoding2, dup
                os.remove(os.path.join(filedir, filename))
            except Exception as e:
                print(cnt, e)
                continue
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from utils import rotateImage
from imagededup.methods import PHash, AHash, CNN, DHash, WHash
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
if __name__ == '__main__':
    rotates = [[], [], [], [], []]
    cnn = CNN()
    for cnt,i in enumerate(paths):
        if cnt%1000==0:
            np.save('imagededup_rotate.npy',rotates)
        for j in range(4):
            filedir = 'benchmark_rotate'
            filename = i[i.rfind('/')+1:]
            try:
                rotateImage(i, filedir, filename)
                # AHash
                phasher = AHash()
                encoding1 = phasher.encode_image(image_file=i)
                encoding2 = phasher.encode_image(image_file=os.path.join(filedir, filename))
                dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                if dup['1']:
                    rotates[0].append(dup['1'][0][1])
                else:
                    rotates[0].append(64)
                # PHash
                phasher = PHash()
                encoding1 = phasher.encode_image(image_file=i)
                encoding2 = phasher.encode_image(image_file=os.path.join(filedir, filename))
                dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                if dup['1']:
                    rotates[1].append(dup['1'][0][1])
                else:
                    rotates[1].append(64)
                # DHash
                phasher = DHash()
                encoding1 = phasher.encode_image(image_file=i)
                encoding2 = phasher.encode_image(image_file=os.path.join(filedir, filename))
                dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                if dup['1']:
                    rotates[2].append(dup['1'][0][1])
                else:
                    rotates[2].append(64)
                # WHash
                phasher = WHash()
                encoding1 = phasher.encode_image(image_file=i)
                encoding2 = phasher.encode_image(image_file=os.path.join(filedir, filename))
                dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                if dup['1']:
                    rotates[3].append(dup['1'][0][1])
                else:
                    rotates[3].append(64)
                # CNN
                encoding1 = cnn.encode_image(image_file=i)
                encoding2 = cnn.encode_image(image_file=os.path.join(filedir, filename))
                dup = cnn.find_duplicates(encoding_map={'1':encoding1[0],'2':encoding2[0]}, min_similarity_threshold=-1.0, scores=True)
                if dup['1']:
                    rotates[4].append(dup['1'][0][1])
                else:
                    rotates[4].append(-1.0)
                del encoding1, encoding2, dup
                os.remove(os.path.join(filedir, filename))
            except Exception as e:
                print(e)
                continue
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from utils import cropImage
from imagededup.methods import PHash, AHash, CNN, DHash, WHash
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
if __name__ == '__main__':
    intra = [[], [], [], [], []]
    cnn = CNN()
    for index, i in enumerate(paths):
        for cnt, j in enumerate(paths[index+1:]):
            if cnt % 1000 == 0:
                np.save('imagededup_intra.npy', intra)
                print(cnt)
            for _ in range(4):
                try:
                    # AHash
                    phasher = AHash()
                    encoding1 = phasher.encode_image(image_file=i)
                    encoding2 = phasher.encode_image(image_file=j)
                    dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                    if dup['1']:
                        intra[0].append(dup['1'][0][1])
                    else:
                        intra[0].append(64)
                    # PHash
                    phasher = PHash()
                    encoding1 = phasher.encode_image(image_file=i)
                    encoding2 = phasher.encode_image(image_file=j)
                    dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                    if dup['1']:
                        intra[1].append(dup['1'][0][1])
                    else:
                        intra[1].append(64)
                    # DHash
                    phasher = DHash()
                    encoding1 = phasher.encode_image(image_file=i)
                    encoding2 = phasher.encode_image(image_file=j)
                    dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                    if dup['1']:
                        intra[2].append(dup['1'][0][1])
                    else:
                        intra[2].append(64)
                    # WHash
                    phasher = WHash()
                    encoding1 = phasher.encode_image(image_file=i)
                    encoding2 = phasher.encode_image(image_file=j)
                    dup = phasher.find_duplicates(encoding_map={'1':encoding1,'2':encoding2}, max_distance_threshold=64, scores=True)
                    if dup['1']:
                        intra[3].append(dup['1'][0][1])
                    else:
                        intra[3].append(64)
                    # CNN
                    encoding1 = cnn.encode_image(image_file=i)
                    encoding2 = cnn.encode_image(image_file=j)
                    dup = cnn.find_duplicates(encoding_map={'1':encoding1[0],'2':encoding2[0]}, min_similarity_threshold=-1.0, scores=True)
                    if dup['1']:
                        intra[4].append(dup['1'][0][1])
                    else:
                        intra[4].append(-1.0)
                    del encoding1, encoding2, dup
                except Exception as e:
                    print('error', e)
                    continue
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from new_utils import cropImage
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
if __name__ == '__main__':
    for cnt, i in enumerate(paths):
        for j in range(4):
            filedir = 'linke_flickr/crop/0-10/'
            filename = str(j)+'_'+i[i.rfind('/')+1:]
            try:
                cropImage(i, filedir, filename, ratio=[0.0, 0.1])
            except Exception as e:
                print(cnt, e)
                continue
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import cv2
import random
import numpy as np
import gzip
import pickle
from new_utils import rotateImage
flickr = './images/'
num = [str(i) for i in range(100)]
paths = []
for i in num:
    _path = os.path.join(flickr, i)
    paths += [os.path.join(_path, j) for j in os.listdir(_path)]
if __name__ == '__main__':
    for cnt, i in enumerate(paths):
        for j in range(4):
            filedir = 'linke_flickr/rotate/0-10/'
            filename = str(j)+'_'+i[i.rfind('/')+1:]
            try:
                rotateImage(i, filedir, filename, rotation=[0,10])
            except Exception as e:
                print(cnt, e)
                continue
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import itertools
import gzip
import pickle
import hashlib
import numpy as np
from utils import mergeVideo, keyFrameExtraction, getImageHashValues, jaccardMeasure, SIFT
import shutil
from eucl_dist.gpu_dist import dist as gdist
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    d = gdist(baseline, vec, optimize_level=3)
    #  d = distance.cdist(baseline, vec, metric)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
roots = ['Moments_in_Time_Mini/training/', 'Moments_in_Time_Mini/validation']
files = []
paths = []
for root in roots:
    dirs = os.listdir(root)
    for d in dirs:
        _dir = os.path.join(root, d)
        _files = os.listdir(_dir)
        files += _files
        paths += [os.path.join(_dir, i) for i in _files]
with gzip.open('mini_video_hash_dict.pkl.gz', 'rb') as f:
    hashes = pickle.load(f)
counter = 0
merge_npy_bench = []
for count,i in enumerate(itertools.combinations(range(len(files)), 2)):
    if count%1000==0:
        np.save('benchmark1.npy',merge_npy_bench)
    video_name = 'merge'+str(counter)+'.mp4'
    keyframedir = 'benchmark_mini_merge/keyframes'
    try:
        mergeVideo(paths[i[0]], paths[i[1]], 'benchmark_mini_merge/video/', video_name)
        keyFrameExtraction(os.path.join( 'benchmark_mini_merge/video/', video_name), keyframedir)
        sift = []
        for j in [os.path.join(keyframedir, s) for s in os.listdir(keyframedir)]:
            _, tmp = SIFT(j)
        #  print(len(tmp))
            for k in tmp:
                sift.append(k)
        sift = np.array(sift)
        #  print(len(sift))
        nn = nearlist_vector(sift)
        h = getImageHashValues(nn)
        con = [paths[i[0]][:paths[i[0]].rfind('/')+1], paths[i[1]][:paths[i[1]].rfind('/')+1]]
        com = [hashlib.md5(files[i[0]].encode('utf8')).hexdigest(), hashlib.md5(files[i[1]].encode('utf8')).hexdigest()]
        merge_npy_bench.append(jaccardMeasure(h,hashes[con[0]+com[0]]))
        merge_npy_bench.append(jaccardMeasure(h,hashes[con[1]+com[1]]))
        shutil.rmtree(keyframedir, ignore_errors=True)
    except Exception as e:
        continue
# coding=utf-8
# Python version: 3.6
from utils import keyFrameExtraction, SIFT
import os
import numpy as np
import base64
import hashlib
source = ['Moments_in_Time_256x256_30fps']
traindir = ['training',  'validation']
for root in source:
    for traintest in traindir:
        _path = os.path.join(root, traintest)
        _pathes = sorted(os.listdir(_path))
        for i in _pathes:
            _files = os.path.join(_path, i)
            print(_files, flush=True)
            try:
                npy_file = 'npy_' + _files
                os.makedirs(npy_file)
            except Exception as e:
                pass
            for j in os.listdir(_files):
                _file = os.path.join(_files, j)
                #  print(_file)
                kf_dir = 'keyframes_' + _file
                os.makedirs(kf_dir)
                keyFrameExtraction(_file, kf_dir)
                npy = []
                for pics in os.listdir(kf_dir):
                    _, des = SIFT(os.path.join(kf_dir, pics))
                    if des is not None:
                        for k in des:
                            npy.append(k)
                if npy:
                    #  np.save(os.path.join(npy_file, base64.encodebytes( j.encode('utf8')).hex())+'.npy', npy)
                     np.save(os.path.join(npy_file, hashlib.md5( j.encode('utf8')).hexdigest())+'.npy', npy)
# coding=utf-8
# Python version: 3.6
import os
import itertools
import gzip
import pickle
import hashlib
import numpy as np
from utils import mergeVideo, keyFrameExtraction, getImageHashValues, jaccardMeasure, SIFT
import shutil
from eucl_dist.gpu_dist import dist as gdist
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    d = gdist(baseline, vec, optimize_level=3)
    #  d = distance.cdist(baseline, vec, metric)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
roots = ['Moments_in_Time_Mini/training/', 'Moments_in_Time_Mini/validation']
files = []
paths = []
for root in roots:
    dirs = os.listdir(root)
    for d in dirs:
        _dir = os.path.join(root, d)
        _files = os.listdir(_dir)
        files += _files
        paths += [os.path.join(_dir, i) for i in _files]
with gzip.open('mini_video_hash.pkl.gz', 'rb') as f:
    hashes = pickle.load(f)
with gzip.open('mini_video_hash_index.pkl.gz', 'rb') as f:
    hashes_index = pickle.load(f)
counter = 0
merge_npy = np.load('merge_video.npy')
for count,i in enumerate(itertools.combinations(range(len(files)), 2)):
    video_name = 'merge'+str(counter)+'.mp4'
    keyframedir = 'mini_merge/keyframes'
    mergeVideo(paths[i[0]], paths[i[1]], 'mini_merge/video/', video_name)
    keyFrameExtraction(os.path.join(
        'mini_merge/video/', video_name), keyframedir)
    sift = []
    for j in [os.path.join(keyframedir, s) for s in os.listdir(keyframedir)]:
        _, tmp = SIFT(j)
        print(len(tmp))
        for k in tmp:
            sift.append(k)
    sift = np.array(sift)
    print(len(sift))
    nn = nearlist_vector(sift)
    h = getImageHashValues(nn)
    result = [i[0:4]+[jaccardMeasure(i[4], h)] for i in hashes]
    sort = sorted(result, key=lambda x: x[4])
    com = [hashlib.md5(files[i[0]].encode('utf8')).hexdigest(
    ), hashlib.md5(files[i[1]].encode('utf8')).hexdigest()]
    if sort[-1][3] == com[0] and sort[-2][3] == com[1]:
        merge_npy.append([counter, sort[-1][4], sort[-2][4]])
        counter += 1
        np.save('merge_video.npy', merge_npy)
    elif sort[-1][3] == com[1] and sort[-2][3] == com[0]:
        merge_npy.append([counter, sort[-1][4], sort[-2][4]])
        counter += 1
        np.save('merge_video.npy', merge_npy)
shutil.rmtree(keyframedir, ignore_errors=True)
# coding=utf-8
# Python version: 3.6
import os
import time
import gzip
import pickle
import numpy as np
from scipy.spatial import distance
from utils import getImageHashValues
from eucl_dist.gpu_dist import dist as gdist
baseline_vector_path = './centroid/0-iteration.npy'
baseline = np.load(baseline_vector_path)
def nearlist_vector(vec, metric='euclidean'):
    """ for each vector in `vec` return the nearlist vector in baseline
    """
    d = gdist(baseline, vec, optimize_level=3)
    #  d = distance.cdist(baseline, vec, metric)
    dargmin = d.argmin(axis=0)
    return baseline[dargmin]
source = ['npy_Moments_in_Time_256x256_30fps']
traindir = ['training',  'validation']
npy = []
for root in source:
    for traintest in traindir:
        _path = os.path.join(root, traintest)
        _pathes = sorted(os.listdir(_path))
        for i in _pathes:
            _files = os.path.join(_path, i)
            print(_files, flush=True)
            try:
                npy_file = 'nn_' + _files
                os.makedirs(npy_file)
            except Exception as e:
                pass
            for j in os.listdir(_files):
                _file = os.path.join(_files, j)
                #  print(_file)
                tmp = np.load(_file)
                nn = nearlist_vector(tmp)
                np.save(os.path.join(npy_file, j), nn)
                npy.append([root, traintest, i, j[:-4],
                            getImageHashValues(nn)])
with gzip.open('30fps_video_hash.pkl.gz', 'wb') as f:
pickle.dump(obj=npy, file=f)
# Python version: 3.6
#!/usr/bin/env python
# coding: utf-8
# Python version: 3.6
# In[1]:
import os
import sqlite3
import cv2
# In[2]:
# In[5]:
flickr = './Image/mirflickr1m/images/'
# In[16]:
def SIFT(src):
    """
    利用cv2图片SIFT获得特征
    :param src: source picture file path
    :return: keypoint and descriptor
    """
    if os.path.isfile(src):
        img = cv2.imread(os.path.abspath(src), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift= cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des
    else:
        logger.error(Fore.RED + src + ': no such file.')
# In[14]:
num = [str(i) for i in range(100)]
# In[18]:
num_of_des = 0
count = 0
for i in num:
    conn = sqlite3.connect('flickr-sift-list.db')
    c = conn.cursor()
    first_dir = os.path.join(flickr, i)
    for j in os.listdir(first_dir):
        if(count%100==0):
            print(i, '---', count)
        if j!='.DS_Store':
            count+=1
            _, des = SIFT(os.path.join(first_dir, j))
            try:
                for k in des:
                    try:
                        c.execute('insert into sift values (?, ?)', (k.tobytes(), i+'/'+j))
                        num_of_des +=1
                    except Exception as exc:
                        if exc == 'UNIQUE constraint failed: sift.des':
                            pass
                        else:
                            print(exc)
            except Exception as exc:
                pass
    conn.commit()
    conn.close()
# In[19]:
print(count)
print(num_of_des)
# In[20]:
j
# In[ ]:
#!/usr/bin/env python
# coding: utf-8
import sqlite3
conn = sqlite3.connect('flickr-sift.db')
c = conn.cursor()
r=c.execute('select count(*) from sift')
for row in r.fetchall():
    print(row)
conn.commit()
conn.close()
#!/usr/bin/env python
# coding: utf-8
# Python version: 3.6
# In[1]:
import os
import sqlite3
import cv2
# In[2]:
# In[5]:
flickr = './Image/mirflickr1m/images/'
# In[16]:
def SIFT(src):
    """
    利用cv2图片SIFT获得特征
    :param src: source picture file path
    :return: keypoint and descriptor
    """
    if os.path.isfile(src):
        img = cv2.imread(os.path.abspath(src), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift= cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des
    else:
        logger.error(Fore.RED + src + ': no such file.')
# In[14]:
num = [str(i) for i in range(100)]
# In[18]:
for i in num:
    conn = sqlite3.connect('flickr-sift.db')
    c = conn.cursor()
    first_dir = os.path.join(flickr, i)
    for j in os.listdir(first_dir):
        print(i, '---', j)
        if j!='.DS_Store':
            _, des = SIFT(os.path.join(first_dir, j))
            try:
                for k in des:
                    try:
                        c.execute('insert into sift values (?, ?)', (k.tobytes(), 1))
                    except Exception as exc:
                        if exc == 'UNIQUE constraint failed: sift.des':
                            pass
                        else:
                            print(exc)
            except Exception as exc:
                pass
    conn.commit()
    conn.close()
# In[19]:
i
# In[20]:
j
# In[ ]:
# coding: utf-8
# Python version: 3.6
import os
import re
   # src 为文件夹的路径，比如 src = './aclImdb/train/neg/10123_3.txt'
   def cleanup(src):
        """
        清除文件里的HTML标签
        :param src: 文件路径
        :return: 清理HTML tag后的文本字符串
        """
        if(os.path.isfile(src)):
            with open(src, 'rb') as f:
                content = f.readlines()[0].decode('utf8') # 经测试所有文件只有一行文本
            cleanr = re.compile('<.*?>')
            return re.sub(cleanr, '', content)
        else:
            raise Exception("File doesn't exist!")
            return False
