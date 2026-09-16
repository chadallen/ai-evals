FROM aisiuk/inspect-tool-support@sha256:fb045da8203aea656785c758f7147b003cfe21f213e9048a38be0a33242a5b3d

# The base image installs Chromium under /root, which the agent cannot traverse.
RUN mv /root/.cache/ms-playwright /opt/ms-playwright \
    && chmod -R a+rX /opt/ms-playwright \
    && useradd --create-home --uid 1000 --shell /bin/bash agent \
    && mkdir /home/agent/workspace \
    && chown agent:agent /home/agent/workspace

ENV HOME=/home/agent \
    PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
WORKDIR /home/agent/workspace
USER agent:agent
