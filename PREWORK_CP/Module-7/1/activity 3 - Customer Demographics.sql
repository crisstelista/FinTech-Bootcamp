-- create database customer_db;
drop table if exists customer;

create table if not exists customer(
	customer_id						int,
	first_name						varchar(50),
	last_name							varchar(50),
	gender								varchar(20),
	age										int,
	address								varchar(125),
	city									varchar(30),
	state									char(2),
	zip_code							int
);

COPY customer(customer_id, first_name, last_name, gender, age, address, city, state, zip_code)
FROM 'D:\FinTech BootCamp\PREWORK_CP\Module-7\Resources\01-Lesson_Plans_07-SQL_1_Activities_07-Stu_Customer_Demographics_Resources_customer.csv'
DELIMITER ','
CSV HEADER;

select * from customer;

select * from customer where gender='Female';

select * from customer where gender='Male' and state='NJ';

select * from customer where gender='Male' and (state='NJ' or state='OH');

select * from customer where gender='Female' and state = 'MD' and age<30;