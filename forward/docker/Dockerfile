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
    git clone https://jayshivpatel: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@github.com/jayshivpatel/M4R.git /tmp/M4R && \
    cd /tmp/M4R

RUN chmod 777 /tmp/M4R

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]