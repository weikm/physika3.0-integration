# build ubuntu mirror
FROM ubuntu:20.04
WORKDIR /
COPY ./ /physika
ENV TZ=UTC
RUN ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime && \
    echo "$TZ" > /etc/timezone && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install cmake -y && \
    apt-get install g++ -y

RUN cd ./physika && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=../../physika_release -DBUILD_SHARED_LIBS=ON -DINSTALL_GTEST=OFF ..&&\
    make && \
    make install && \
    rm -r /physika

FROM ubuntu:20.04
WORKDIR /
COPY --from=0 /physika_release ./physika_release