import urllib
import requests

# 禁用安全请求警告
requests.packages.urllib3.disable_warnings()

url = 'https://kesci-datasets-ng.s3.cn-northwest-1.amazonaws.com.cn/5d09b973921a91002b8dd873/1561011603429_1/cs-test.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAO66SICIVCILDTITQ%2F20210718%2Fcn-northwest-1%2Fs3%2Faws4_request&X-Amz-Date=20210718T135934Z&X-Amz-Expires=7200&X-Amz-Signature=191217e2ff8408ac76d1e28d5b47845429ad4bcf7f7490050a843d60b80cec07&X-Amz-SignedHeaders=host'
header = { # 头部伪装
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36 Edg/91.0.864.59"}

r = requests.get(url, headers=header)

with open("data.csv", "wb") as code:
    code.write(r.content)

urllib.request.urlretrieve(url, "data.csv")
print("文件爬取成功")

import wget

#url = 'https://kesci-datasets-ng.s3.cn-northwest-1.amazonaws.com.cn/5d09b973921a91002b8dd873/1561011603429_1/cs-test.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAO66SICIVCILDTITQ%2F20210718%2Fcn-northwest-1%2Fs3%2Faws4_request&X-Amz-Date=20210718T135934Z&X-Amz-Expires=7200&X-Amz-Signature=191217e2ff8408ac76d1e28d5b47845429ad4bcf7f7490050a843d60b80cec07&X-Amz-SignedHeaders=host'
url ='https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/2551/29345/compressed/cs-training.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1626889435&Signature=ZWspfyAn1MTF9Tl7aGEWYIAc%2FLO0bhXLUXGOozZkXNah1mSnIfCcPdjTw6Nt496Etg9p8RpHrh%2BcI2iuJYXkkf1plaChoJF36nFqfaxgzvgmjTN41nD1g4oDo846H7goBMchBmgn%2BplJFU70RZgPF0EsQYlR4MGQxoSj01dNNp5TmQrNYFGD%2Faa2h9AYvgs2ttweYZ3Os2osVChMEOR%2F6vu%2Fqndnax0aX2X2GO6cKdmiwyDTdKZg3t50B9g8JtflWWqkwN51SZETu1IqGUxyPpE6bYXQvQ43VfIDwIcMcI9DPspE%2FRCh6MHSDpKMccx0BmZpbDW8tknw1L0OJJUyPg%3D%3D&response-content-disposition=attachment%3B+filename%3Dcs-training.csv.zip'
wget.download(url,'data.zip',)
print("文件爬取成功")



