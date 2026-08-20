-- Tự động tạo database và user cho Feast registry
-- Script này được postgres Docker image chạy tự động khi khởi tạo lần đầu
-- (nhờ mount vào /docker-entrypoint-initdb.d/)

CREATE DATABASE feast;
CREATE USER airflow WITH PASSWORD 'airflow';
GRANT ALL PRIVILEGES ON DATABASE feast TO airflow;
\c feast
GRANT ALL ON SCHEMA public TO airflow;