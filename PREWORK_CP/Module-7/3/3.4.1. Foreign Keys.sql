drop table if exists estates_new;
drop table if exists estate_type;
drop table if exists estates;
drop table if exists owners;

create table owners (
	owner_id		SERIAL PRIMARY KEY,
	first_name	varchar(40),
	last_name		varchar(40)
);

create table estates(
	estate_id		SERIAL PRIMARY KEY,
	owner_id		INT,
	address			varchar(50),
	city				varchar(20),
	state				char(2),
	zip_code		char(5),
	CONSTRAINT fk_estates_owners FOREIGN KEY(owner_id) REFERENCES owners(owner_id)
);

create table estate_type(
	estate_type_id	SERIAL PRIMARY KEY,
	estate_type			varchar(30)
);


create table estates_new(
	estate_id				INT,
	owner_id				INT,
	address					varchar(50),
	city						varchar(20),
	state						char(2),
	zip_code				char(5),
	estate_type_id	INT,
	CONSTRAINT fk_estates_new_owners FOREIGN KEY(owner_id) REFERENCES owners(owner_id),
	CONSTRAINT fk_estates_new_estate_type_id FOREIGN KEY(estate_type_id) REFERENCES estate_type(estate_type_id)
);


select * from estates_new;
select * from estate_type;
select * from estates;
select * from owners;

select * 
from owners o
join estates e on o.owner_id = e.owner_id;

select *
from owners o
join estates_new e on o.owner_id = e.owner_id
join estate_type et on et.estate_type_id = e.estate_type_id
;
