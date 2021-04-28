-- create database bank_db;
drop table if exists banks;

create table if not exists  banks(
	bank_id								SERIAL PRIMARY KEY,
	bank_name							varchar(50),
	bank_routing_number		BIGINT
);

insert into banks(bank_name, bank_routing_number)
values 
('Bank of America', 198491827),
('Wells Fargo', 629873495),
('JPMorgan Chase', 2340903984),
('Citigroup', 890123900),
('TD Bank', 905192010),
('Capital One', 184619239),
('Capital One', 184619239);

select * from banks;
select * from banks where bank_name = 'Capital One';

insert into banks(bank_name, bank_routing_number)
values
('My first bank', 123456789),
('My second bank', 234567890),
('Ally Bank', 316289502),
('Discover Bank', 639893944),
('Bank of New York Mellon', 8734569384);

select * from banks;

update banks set
	bank_name = 'PNC Bank'
where bank_name = 'Citigroup';

select * from banks;

update banks set
	bank_routing_number = 1995826182
where bank_name = 'Wells Fargo';

select * from banks;

alter table banks add column mortgage_lending BOOLEAN default TRUE;

select * from banks;

