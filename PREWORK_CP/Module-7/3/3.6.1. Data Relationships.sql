drop table if exists agent_region_junction;
drop table if exists regions;
drop table if exists agents;

create table agents (
	agent_id			serial primary key, 
	first_name		varchar(50), 
	last_name			varchar(50)
);

create table regions (
	region_id			serial primary key, 
	region_name		varchar(50)
);

create table agent_region_junction(
	agent_id			int,
	region_id			int,
	PRIMARY KEY(agent_id, region_id),
	CONSTRAINT fk_agent_region_junction_agents FOREIGN KEY(agent_id) REFERENCES agents(agent_id),
	CONSTRAINT fk_agent_region_junction_regions FOREIGN KEY(region_id) REFERENCES regions(region_id)
);



