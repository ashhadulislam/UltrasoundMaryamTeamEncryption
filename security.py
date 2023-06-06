import pyaes, pbkdf2, binascii, os, secrets
import pickle


def generate_key():
    if not os.path.isfile("key.pkl"):
        # Derive a 256-bit AES encryption key from the password
        password = "s3cr3t*c0d3"
        passwordSalt = os.urandom(16)
        key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
        # open a file, where you ant to store the data
        file = open('key.pkl', 'wb')

        # dump information to that file
        pickle.dump(key, file)
        # close the file
        file.close()            
    else:
        # open a file, where you stored the pickled data
        file = open('key.pkl', 'rb')
        # dump information to that file
        key = pickle.load(file)

        # close the file
        file.close()    
    return key    

def generate_iv():
    if not os.path.isfile("iv.pkl"):
        iv = secrets.randbits(256)
        file = open('iv.pkl', 'wb')
        pickle.dump(iv, file)
        file.close()
    else:
        file = open('iv.pkl', 'rb')
        iv = pickle.load(file)
        file.close()    
    return iv



def encrypt(plaintext="This is a secret message"):

    key=generate_key()

    # Encrypt the plaintext with the given key:
    # ciphertext = AES-256-CTR-Encrypt(plaintext, key, iv)
    iv = generate_iv()

    print("key:",key)
    print("iv:",iv)


    
    aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
    ciphertext = aes.encrypt(plaintext)
    print("Enc:",ciphertext)
    print("Enc str:",str(ciphertext))
    print('Encrypted hexlify:', binascii.hexlify(ciphertext))
    encr=binascii.hexlify(ciphertext)
    encr=encr.decode("utf-8")
    print('Encrypted hexlify str:', encr)    
    return encr





def decrypt(ciphertext='da15f3a7288dd0c79a47435a00f6121625a6e1fcf6be997a'):
    key=generate_key()
    iv=generate_iv()
    ciphertext=binascii.unhexlify(ciphertext)
    aes = pyaes.AESModeOfOperationCTR(key, pyaes.Counter(iv))
    decrypted = aes.decrypt(ciphertext)
    print(decrypted)


