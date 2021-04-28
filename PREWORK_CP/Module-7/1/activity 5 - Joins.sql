-- create database payments_db;

drop table if exists payments;
drop table if exists banks;

create table if not exists payments(
	payment_id						SERIAL PRIMARY KEY,
	bank_number 					bigint,
	bank_routing_number 	bigint,
	customer_id						int
);

create table if not exists  banks(
	bank_id								SERIAL PRIMARY KEY,
	bank_name							varchar(50),
	bank_routing_number		BIGINT
);

COPY payments(payment_id,bank_number, bank_routing_number, customer_id)
FROM 'D:\FinTech BootCamp\PREWORK_CP\Module-7\Resources\01-Lesson_Plans_07-SQL_1_Activities_10-Stu_Joins_Resources_payments.csv'
DELIMITER ','
CSV HEADER;

COPY banks(bank_id, bank_name, bank_routing_number)
FROM 'D:\FinTech BootCamp\PREWORK_CP\Module-7\Resources\01-Lesson_Plans_07-SQL_1_Activities_10-Stu_Joins_Resources_banks.csv'
DELIMITER ','
CSV HEADER;

select * from payments;
select * from banks;

select * 
from payments p 
join banks b on p.bank_routing_number = b.bank_routing_number;

select * 
from payments p 
left outer join banks b on p.bank_routing_number = b.bank_routing_number;

select * 
from payments p 
right outer join banks b on p.bank_routing_number = b.bank_routing_number;

select * 
from payments p
full outer join banks b on p.bank_routing_number = b.bank_routing_number;

select * 
from payments p
cross join banks b;

