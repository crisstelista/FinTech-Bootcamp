-- Create a new table first_nf_employee and that organizes the data in employee_normalization according to first normal form standards.
create table forms_address (
	address_id		serial primary key,
	address				varchar(128),
	city					varchar(30),
	state					char(2),
	zip_code			char(5)
);


create table forms_employee (
	employee_id		serial primary key,
	name					varchar(128),
	age						int
);

create table first_nf_employee (
		address_id			serial,
		employee_id			serial
);



-- Then, create two new tables second_nf_employee and second_nf_employee_email that organizes the data in first_nf_employee according to second normal form standards.
create table second_nf_employee (
	employee_id		serial primary key,
	name					varchar(128),
	age						int
);

create table second_nf_email (
	employee_id		serial primary key,
	email					varchar(128)
);

create table second_nf_employee (
		address_id			serial,
		employee_id			serial
);

-- Lastly, create two new tables third_nf_employee and third_nf_zipcode that organizes the data in second_nf_employee according to third normal form standards.

