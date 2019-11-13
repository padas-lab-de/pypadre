Installing and configuring a local gitlab serve instance for testing purposes (using docker-compose):
1. create a manifest file "docker-compose.yml" that contains the following:
"""
version : '3.3'
services:
  gitlab:
    image: 'gitlab/gitlab-ce:latest'
    container_name : gitlab
    restart: always
    hostname: 'gitlab.padre.backend'
#    volumes:
#      - data/gitlab-config:/etc/gitlab:rw
#      - data/gitlab-logs:/var/log/gitlab:rw
#      - data/gitlab-data:/var/opt/gitlab:rw
    links:
      - postgresql:postgresql
      - redis:redis
    environment:
      GITLAB_OMNIBUS_CONFIG: |
        postgresql['enable'] = false
        gitlab_rails['db_username'] = "gitlab"
        gitlab_rails['db_password'] = "gitlab"
        gitlab_rails['db_host'] = "postgresql"
        gitlab_rails['db_port'] = "5432"
        gitlab_rails['db_database'] = "gitlab_db"
        gitlab_rails['db_adapter'] = 'postgresql'
        gitlab_rails['db_encoding'] = 'utf8'
        redis['enable'] = false
        gitlab_rails['redis_host'] = 'redis'
        gitlab_rails['redis_port'] = '6379'
        external_url 'http://gitlab.padre.backend:30080'
        gitlab_rails['gitlab_shell_ssh_port'] = 30022
    ports:
      - "30080:30080"
      - "30022:22"
    networks:
      - gitlab-net


  postgresql:
    restart: always
    image: postgres:alpine
    container_name : postgresql
    environment:
      - POSTGRES_USER=gitlab
      - POSTGRES_PASSWORD=gitlab
      - POSTGRES_DB=gitlab_db
#    volumes:
#      - data/postgresql:/var/lib/postgresql:rw
    networks:
      - gitlab-net

  redis:
    restart: always
    image: redis:alpine
    container_name : redis
    networks:
      - gitlab-net

networks:
  gitlab-net:

"""

2. Run : "docker-compose up -d" under the same folder of the created yaml file.
3. Add "127.0.0.1 gitlab.padre.backend" to your etc/hosts/file.
4. Access http://gitlab.padre.backend:30080/ and change the root user password
5. Register an account and create an access token
6. Use the gitlab_url "gitlab.padre.backend:30080" and token that you generated in your padre config file:

"""
[GITLAB BACKEND]
root_dir = your root directory
gitlab_url = "http:gitlab.padre.backend:30080/" (in this case)
user = "username" of your account
token = "generated_token"
"""
