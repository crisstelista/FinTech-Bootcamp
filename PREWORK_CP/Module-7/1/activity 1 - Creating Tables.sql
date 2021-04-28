-- CREATE DATABASE state_info;

DROP TABLE IF EXISTS states;

CREATE TABLE IF NOT EXISTS states(
	state_name							varchar(30),
	state_abbr							char(2),
	population							int,
	state_property_tax_rate	FLOAT
);

insert into states (state_name, state_abbr, population)
values 
('Florida', 'FL', 21477737);

alter table states add COLUMN one_col varchar(20);

select * from states;

insert into states (state_name, state_abbr, population, state_property_tax_rate)
values 
('Alabama', 'AL', 4903185, 0.0042),
('Texas', 'TX', 28995881, 0.0181),
('Kentucky', 'KY', 4467673, 0.0086),
('Virginia', 'VA', 8535519, 0.0081),
('Louisiana', 'LA', 4648794, 0.0053),
('Utah', 'UT', 3205958, 0.0064),
('Vermont', 'VT', 623989, 0.0188);

update states set 
	state_property_tax_rate = 0.0093
where state_abbr='FL';

select state_name from states;

select state_abbr from states;

select * from states where population > 5000000;

select * from states where population > 5000000 and state_property_tax_rate<0.01;

