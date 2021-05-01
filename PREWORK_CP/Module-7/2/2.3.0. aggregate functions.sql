--1. What is the average payment amount?
select round(avg(amount),2)
from payment;

--2. What is the total payment amount?
select sum(amount)
from payment;

--3. What is the minimum payment amount?
select min(amount)
from payment;

--4. What is the maximum payment amount?
select max(amount)
from payment;

--5. How many customers has each staff serviced?
select count(customer_id), staff_id
from payment
group by staff_id
;
--6. What is the count of payments for each customer?
select count(payment_id), customer_id
from payment
group by customer_id
;

--7. Which customers have made over 40 payments?
select count(payment_id), customer_id
from payment
group by customer_id
having count(payment_id)>40
;
