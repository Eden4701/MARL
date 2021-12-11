import tenseal as ts
import numpy as np
# Setup TenSEAL context
import torch
# def enc1(in_t):
#     plain_vector = ts.plain_tensor(in_t)
#
#     encrypted_vector = ts.ckks_vector(context, plain_vector)
#
#     print(encrypted_vector.decrypt().tolist())  # tolist是用来将多项式m(X)解码成张量的
#
#     # 加法和乘法直接使用+, *, @ 即可
#     m1 = [1, 2, 3]
#     m2 = [4, 5, 6]
#     p1, p2 = ts.plain_tensor(m1), ts.plain_tensor(m2)
#     e1, e2 = ts.ckks_vector(context, p1), ts.ckks_vector(context, p2)


def enc(a_list):
    enc_next_obs=[]
    for i in range(0,len(a_list)):
        a_list[i]=a_list[i].tolist()
        print("a_list[i]",a_list[i][0])
        enc=ts.ckks_vector(context, a_list[i][0])
        print("enc",enc)
        enc_next_obs.append(enc)
    return enc_next_obs
poly_mod_degree = 32768
coeff_mod_bit_sizes = [31,26,26,26,26,26,26,31]
ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_training.global_scale = 2 ** 21
ctx_training.generate_galois_keys()
"向量方式加密"
def fl_enc(x_train):
    enc_x_train=[]
    # for i in range (0,len(x_train)):
    #     enc_x_train=ts.ckks_vector(ctx_training, x_train[i].tolist())

    enc_x_train = [ts.ckks_vector(ctx_training, x.tolist()) for x in x_train]
    print("fl_enc",len(enc_x_train))
    # a=[]
    # for i in range(0,69):
    #     a.append(2)
    # enc_x_train[0].dot(a)
    #print(x_train[0].tolist() * 2)
    # print(enc_x_train[0].decrypt())
    return enc_x_train
"卷积方式的加密"
def convolution_enc( plain_input):
    print(plain_input)
    # kernal size:64*64 , 步长：64
    enc_input, windows_nb = ts.im2col_encoding(context, plain_input.tolist(), 64, 64, 64)#Encoding an image into a CKKSVector
    #assert windows_nb == 64
    print("windows_nb",windows_nb)#卷积块数量
    print("enc_input",enc_input)
    return enc_input
def enc1(x_train):
    print("x_train", len(x_train))
    # for i in range (0,len(x_train)):
    enc_x_train=ts.ckks_vector(ctx_training, x_train)
    #enc_x_train = [ts.ckks_vector(ctx_training, x.tolist()) for x in x_train]
    return enc_x_train
#  Context object, holding the encryption parameters and keys
def context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, 32768, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context

context = context()
import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_PKC
from Crypto import Random
from Crypto.Hash import SHA256
from Crypto.Signature import PKCS1_v1_5 as Signature_PKC

# def create_rsa_key():
#     """
#
#     创建RSA密钥
#
#     步骤说明：
#
#     1、从 Crypto.PublicKey 包中导入 RSA，创建一个密码
#
#     2、生成 1024/2048 位的 RSA 密钥
#
#     3、调用 RSA 密钥实例的 exportKey 方法，传入密码、使用的 PKCS 标准以及加密方案这三个参数。
#
#     4、将私钥写入磁盘的文件。
#
#     5、使用方法链调用 publickey 和 exportKey 方法生成公钥，写入磁盘上的文件。
#
#     """
#     # 伪随机数生成器6
#     random_gen = Random.new().read
#     # 生成秘钥对实例对象：1024是秘钥的长度
#     rsa = RSA.generate(1024, random_gen)
#
#     # Server的秘钥对的生成
#     private_pem = rsa.exportKey()
#     with open("server_private.pem", "wb") as f:
#         f.write(private_pem)
#
#     public_pem = rsa.publickey().exportKey()
#     with open("server_public.pem", "wb") as f:
#         f.write(public_pem)
#
#     # Client的秘钥对的生成
#     private_pem = rsa.exportKey()
#     with open("client_private.pem", "wb") as f:
#         f.write(private_pem)
#
#     public_pem = rsa.publickey().exportKey()
#     with open("client_public.pem", "wb") as f:
#         f.write(public_pem)

class HandleRSA():
    def create_rsa_key(self):
        """

        创建RSA密钥

        步骤说明：

        1、从 Crypto.PublicKey 包中导入 RSA，创建一个密码

        2、生成 1024/2048 位的 RSA 密钥

        3、调用 RSA 密钥实例的 exportKey 方法，传入密码、使用的 PKCS 标准以及加密方案这三个参数。

        4、将私钥写入磁盘的文件。

        5、使用方法链调用 publickey 和 exportKey 方法生成公钥，写入磁盘上的文件。

        """
        # 伪随机数生成器
        random_gen = Random.new().read
        # 生成秘钥对实例对象：1024是秘钥的长度
        rsa = RSA.generate(1024, random_gen)

        # Server的秘钥对的生成
        private_pem = rsa.exportKey()
        with open("server_private.pem", "wb") as f:
            f.write(private_pem)

        public_pem = rsa.publickey().exportKey()
        with open("server_public.pem", "wb") as f:
            f.write(public_pem)

        # Client的秘钥对的生成
        private_pem = rsa.exportKey()
        with open("client_private.pem", "wb") as f:
            f.write(private_pem)

        public_pem = rsa.publickey().exportKey()
        with open("client_public.pem", "wb") as f:
            f.write(public_pem)

    # Server使用Client的公钥对内容进行rsa 加密
    def encrypt(self, plaintext):
        """
        client 公钥进行加密
        plaintext:需要加密的明文文本，公钥加密，私钥解密
        """

        # 加载公钥
        rsa_key = RSA.import_key(open("client_public.pem").read() )

        # 加密
        cipher_rsa = Cipher_PKC.new(rsa_key)
        en_data = cipher_rsa.encrypt(plaintext.encode("utf-8")) # 加密

        # base64 进行编码
        base64_text = base64.b64encode(en_data)

        return base64_text.decode() # 返回字符串

    # Client使用自己的私钥对内容进行rsa 解密
    def decrypt(self, en_data):
        """
        en_data:加密过后的数据，传进来是一个字符串
        """
        # base64 解码
        base64_data = base64.b64decode(en_data.encode("utf-8"))

        # 读取私钥
        private_key = RSA.import_key(open("client_private.pem").read())

        # 解密
        cipher_rsa = Cipher_PKC.new(private_key)
        data = cipher_rsa.decrypt(base64_data,None)

        return data.decode()

    # Server使用自己的私钥对内容进行签名
    def signature(self,data:str):
        """
         RSA私钥签名
        :param data: 明文数据
        :return: 签名后的字符串sign
        """

        # 读取私钥
        private_key = RSA.import_key(open("server_private.pem").read())
        # 根据SHA256算法处理签名内容data
        sha_data= SHA256.new(data.encode("utf-8")) # b类型

        # 私钥进行签名
        signer = Signature_PKC.new(private_key)
        sign = signer.sign(sha_data)

        # 将签名后的内容，转换为base64编码
        sign_base64 = base64.b64encode(sign)
        return sign_base64.decode()

    # Client使用Server的公钥对内容进行验签

    def verify(self,data:str,signature:str) -> bool:
        """
        RSA公钥验签
        :param data: 明文数据,签名之前的数据
        :param signature: 接收到的sign签名
        :return: 验签结果,布尔值
        """
        # 接收到的sign签名 base64解码
        sign_data = base64.b64decode(signature.encode("utf-8"))

        # 加载公钥
        piblic_key = RSA.importKey(open("server_public.pem").read())

        # 根据SHA256算法处理签名之前内容data
        sha_data = SHA256.new(data.encode("utf-8"))  # b类型

        # 验证签名
        signer = Signature_PKC.new(piblic_key)
        is_verify = signer.verify(sha_data, sign_data)

        return is_verify







v=[1,1,2]
v2=[1,2,0]
v3=[1,2,3]
v4=[5,6,7]
def a():

    v1=ts.ckks_vector(ctx_training,v)
    a=v1.dot([1,3.1,5.1])#[16.38506430834864] 不准
    print(a.decrypt())

def randomization_matrix(m,n):
   x = np.random.random([m,n])
   return x



def b():
    p=[]
    p.append(ts.ckks_vector(ctx_training,v))
    p.append(ts.ckks_vector(ctx_training,v2))
    p.append(ts.ckks_vector(ctx_training, v3))
    p.append(ts.ckks_vector(ctx_training, v4))
    out=ts.pack_vectors(p)
    random_mat = randomization_matrix(256, 64)
    aaa=out.mm_(random_mat.tolist())
    print(aaa.decrpyt())


if __name__ == '__main__':
    b()


v1 = [0, 1, 2, 3, 4]
v2 = [4, 3, 2, 1, 0]

# encrypted vectors
enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

result = enc_v1 + enc_v2
result.decrypt() # ~ [4, 4, 4, 4, 4]

result = enc_v1.dot(enc_v2)
result.decrypt() # ~ [10]

matrix = [
  [73, 0.5, 8],
  [81, -5, 66],
  [-100, -78, -2],
  [0, 9, 17],
  [69, 11 , 10],
]
result = enc_v1.matmul(matrix)
result.decrypt() # ~ [157, -90, 153]