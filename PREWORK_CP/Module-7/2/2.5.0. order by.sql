-- Find the count of payments per customer in descending order
select count(payment_id), customer_id
from payment
group by customer_id
order by count(payment_id) desc
;

-- Find the top 5 customers who have spend the most money
select sum(amount), customer_id
from payment
group by customer_id
order by sum(amount) desc
limit 5
;

-- Find the bottom 5 customers who have spend the least money
select sum(amount), customer_id
from payment
group by customer_id
order by sum(amount) asc
limit 5
;

-- Find the top 10 customers with the highest average payment
-- rounded to two decimal places
select round(avg(amount),2), customer_id
from payment
group by customer_id
order by avg(amount) desc
limit 10
;

-- BONUS 1
-- Find the staff names and their number of customers serviced in descending order.
select count(customer_id), concat(s.first_name, ', ', s.last_name) as "Staff Name"
from payment p
join staff s on p.staff_id = s.staff_id
group by s.staff_id
order by count(customer_id) desc;

-- BONUS 2
-- Using the CAST() function, cast the payment_date as a DATE datatype to group by day (rather than date and time). Determine the count of payments per day in descending order. Read more here.
select count(payment_id) as "Number of Payments", EXTRACT(DAY FROM payment_date) as "Day of Month"
from payment
group by EXTRACT(DAY FROM payment_date)
order by count(payment_id) desc;

