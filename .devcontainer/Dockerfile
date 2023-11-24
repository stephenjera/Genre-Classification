FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Copy environment.yaml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yaml exists.
COPY environment.yaml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yaml" ]; then umask 0002 \
    && /opt/conda/bin/conda env create -f /tmp/conda-tmp/environment.yaml -y; fi \
    && rm -rf /tmp/conda-tmp

USER vscode
RUN git clone https://github.com/spaceship-prompt/spaceship-prompt.git /home/vscode/.oh-my-zsh/custom/themes/spaceship-prompt --depth=1 \ 
    && ln -s /home/vscode/.oh-my-zsh/custom/themes/spaceship-prompt/spaceship.zsh-theme /home/vscode/.oh-my-zsh/custom/themes/spaceship.zsh-theme \
    && sed -i 's/^ZSH_THEME=.*/ZSH_THEME="spaceship"/' /home/vscode/.zshrc
