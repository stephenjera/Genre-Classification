FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Copy environment.yaml to a temp location so we update the environment. 
COPY environment.yaml /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yaml" ]; then umask 0002 \
    && /opt/conda/bin/conda env create -f /tmp/conda-tmp/environment.yaml -y; fi \
    && rm -rf /tmp/conda-tmp

# Customise terminal 
USER vscode
RUN git clone https://github.com/spaceship-prompt/spaceship-prompt.git /home/vscode/.oh-my-zsh/custom/themes/spaceship-prompt --depth=1 \ 
    && ln -s /home/vscode/.oh-my-zsh/custom/themes/spaceship-prompt/spaceship.zsh-theme /home/vscode/.oh-my-zsh/custom/themes/spaceship.zsh-theme \
    && sed -i 's/^ZSH_THEME=.*/ZSH_THEME="spaceship"/' /home/vscode/.zshrc \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && sed -i 's/plugins=(git)/plugins=(git z zsh-syntax-highlighting zsh-autosuggestions)/g' ~/.zshrc
RUN bash -c 'conda init bash && source ~/.bashrc' && zsh -c 'conda init zsh && source ~/.zshrc'   