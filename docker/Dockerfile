FROM julia:latest

RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    ssh-askpass \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

COPY id_rsa.pub /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

RUN rm -rf /tmp/M4R && \
    git clone https://github.com/JayshivPatel/FastRecurrenceArrays.git /tmp/FastRecurrenceArrays && \
    cd /tmp/FastRecurrenceArrays

RUN chmod 777 /tmp/FastRecurrenceArrays

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]