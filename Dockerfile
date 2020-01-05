FROM centos
MAINTAINER DangPham

# Utilities
RUN yum install -y wget bzip2 gcc
RUN yum install -y epel-release
RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm
RUN yum -y update

# unzip package
RUN yum -y install unzip

# Dirs
RUN mkdir /source

# Python3 and Pip
RUN yum install -y python36u python36u-libs python36u-devel python36u-pip

RUN pip3.6 install --upgrade pip
RUN pip3.6 install setuptools  --upgrade
RUN yum install -y pylint

# Gcc
RUN yum -y install gcc
RUN yum -y install gcc-c++


# install pip 3
RUN pip3.6 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl

RUN mkdir -p /source/ner
ADD requirements.txt /source/ner

# Install requirements
RUN pip3.6 install -r /source/ner/requirements.txt

# Copy sources
ADD ./ /source/ner
WORKDIR /source/ner


