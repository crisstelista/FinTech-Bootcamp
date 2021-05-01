-- Create customer revenue view
-- Write a query to get the number of payments and total payment amount for each customer in the payment table. Make sure to pull the customer first_name and last_name from the customer table via a JOIN.
-- Then, create a view named customer_revenues from the above query.
-- Query the newly created customer_revenues view to find the revenues associated with customer 'THERESA WATSON'.
DROP VIEW IF EXISTS CUSTOMER_REVENUES ;

create view customer_revenues as 
select 	count(p.payment_id) as "number of payments",
				sum(p.amount) as "Total Payment Amount",
				c.first_name as "Customer First Name",
				c.last_name as "Customer Last Name"
from payment p
join customer c on p.customer_id = c.customer_id
GROUP BY c.first_name, last_name
order by sum(amount) desc;			
				
				
select * 
from customer_revenues
where "Customer First Name" = 'THERESA'
and "Customer Last Name" = 'WATSON'


-- Query customer revenue view
-- Write a query to get the number of payments and total payment amount that the staff member 'Mike Hillyer' has facilitated for each day in the payment table. Make sure to use a subquery instead of a join to do so.
-- Query the newly created staff_sales view to find the sales for staff_id = 1 on the date 2005-07-31.


-- BONUS
-- Create staff sales view


-- Query staff sales view
