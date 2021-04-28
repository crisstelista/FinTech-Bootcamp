-- create database sales_db;

drop table if exists sales;

create table if not exists sales(
	sale_id			SERIAL PRIMARY KEY,
	payment_id 		int,
	mortgage_id 	int,
	loan_amount		float,
	loan_date			date
);

COPY sales(sale_id,payment_id,mortgage_id,loan_amount,loan_date)
FROM 'D:\FinTech BootCamp\PREWORK_CP\Module-7\Resources\01-Lesson_Plans_07-SQL_1_Activities_08-Stu_CRUD_Resources_sales.csv'
DELIMITER ','
CSV HEADER;

select * from sales;

select * from sales where loan_amount < 300000;

select avg(loan_amount) from sales;

update sales set 
	loan_amount = 423212
where sale_id = 33;

select * from sales;

alter table sales add column loan_distributed boolean default TRUE;

select * from sales;

insert into sales (sale_id, payment_id, mortgage_id, loan_amount, loan_date)
values (101,101,2,734544, '1995-10-05');

select * from sales;

delete from sales where sale_id = 72;

select * from sales;
