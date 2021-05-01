-- Using subqueries, display the titles of all films of which
-- employee `Jon Stephens` rented out to customers.

select f.title
from film f
join inventory i on i.film_id = f.film_id
join rental r on r.inventory_id = i.inventory_id
join payment p on p.rental_id = r.rental_id
join staff s on s.staff_id = p.staff_id
where s.first_name = 'Jon'
and s.last_name = 'Stephens'
group by f.film_id
order by f.title asc
;

select title 
from film
where film_id in (
		select film_id
		from inventory
		where inventory_id in (
				select inventory_id
				from rental
				where rental_id in (
						select rental_id
						from payment
						where staff_id in (
								select staff_id
								from staff s
								where s.first_name = 'Jon'
								and s.last_name = 'Stephens'
								)
						)
				)
		)
order by title asc
;
-- Using subqueries, find the total rental amount paid for the film `ACE GOLDFINGER`
select sum(amount)
from film f
join inventory i on i.film_id = f.film_id
join rental r on r.inventory_id = i.inventory_id
join payment p on p.rental_id = r.rental_id
join staff s on s.staff_id = p.staff_id
where f.title = 'ACE GOLDFINGER'
group by f.film_id
order by f.title asc
;

SELECT sum(amount)
FROM payment 
where rental_id in 
		(select rental_id 
		from rental 
		where inventory_id in (
				select inventory_id 
				from inventory
				where film_id in (
						select film_id
						from film
						where title ='ACE GOLDFINGER'
						)
				)
		)
;









