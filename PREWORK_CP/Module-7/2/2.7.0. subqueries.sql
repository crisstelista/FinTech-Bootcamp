-- 1. Find the customer first and last names of those who have made payments.
select c.first_name, c.last_name, p.payment_date, p.amount
from customer c 
join payment p on c.customer_id = p.payment_id
where p.amount > 0
;

select c.first_name, c.last_name, p.payment_date
from customer c 
where c.customer_id in (select customer_id from payment p)
;








-- 2. Find the staff email addresses of those who have helped customers make payments.
select s.email, count(payment_id)
from payment p 
join staff s on p.staff_id = s.staff_id
group by s.staff_id
having count(payment_id)>0

-- 3. Find the rental records of all films that have been rented out and paid for.
select r.*
from rental r
join payment p on r.rental_id = p.rental_id
where p.amount>0
;

-- BONUS
-- Use the payment, rental, inventory, and film tables to find the titles of all films that have been rented out and paid for.
select distinct f.title
from payment p 
join rental r on r.rental_id = p.payment_id
join inventory i on i.inventory_id = r.inventory_id
join film f on f.film_id = i.film_id
where p.amount>0
order by f.title asc;



