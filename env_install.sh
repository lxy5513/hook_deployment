apt update -y
apt install python3-distutils -y
apt install vim -y
apt-get install -y libsm6 libxext6 libxrender-dev
apt-get install libglib2.0-0 -y
pip install pip -U
pip install -i https://pypi.douban.com/simple/ pip -U
pip config set global.index-url https://pypi.douban.com/simple/
apt-get install openjdk-11-jre
pip install torchserve torch-model-archiver

keytool -genkey -keyalg RSA -alias ts -keystore keystore.p12 -storepass changeit -storetype PKCS12 -validity 3600 -keysize 2048 -dname "CN=www.MY_TS.com, OU=Cloud Service, O=model server, L=Palo Alto, ST=California, C=US"
