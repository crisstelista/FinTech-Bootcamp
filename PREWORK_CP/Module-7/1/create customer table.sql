CREATE TABLE IF NOT EXISTS customer (
	id	SERIAL PRIMARY KEY,
	first_name varchar(20) not null,
	last_name varchar(20),
	gender varchar(20),
	age int,
	address varchar(50),
	city varchar(30),
	state char(2),
	zipcode char(5)	
);

insert into customer (first_name, last_name, gender, age, address, city, state, zipcode)
values 
('Michael', 'Meyer', 'Male', 24, 'address 1', 'Astoria', 'NY', '11105'),
('John', 'Smith', 'Male', 12, 'address 2', 'Whitestone', 'NY', '11357');

select * FROM customer;

select * 
from customer 
where age>=12
and first_name = 'John';
	