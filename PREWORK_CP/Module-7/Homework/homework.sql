CREATE TABLE "card_holder" (
  "id" int4 NOT NULL,
  "name" varchar(255),
  PRIMARY KEY ("id")
);

CREATE TABLE "credit_card" (
  "card" VARCHAR(20) NOT NULL,
  "card_holder_id" int4,
  PRIMARY KEY ("card")
);

CREATE TABLE "merchant" (
  "id" int4 NOT NULL,
  "name" varchar(255),
  "merchant_category_id" int4,
  PRIMARY KEY ("id")
);

CREATE TABLE "merchant_category" (
  "id" int4 NOT NULL,
  "name" varchar(255),
  PRIMARY KEY ("id")
);

CREATE TABLE "transaction" (
  "id" int4 NOT NULL,
  "date" timestamp,
  "amount" float8,
  "card" VARCHAR(20),
  "id_merchant" int4,
  PRIMARY KEY ("id")
);

ALTER TABLE "credit_card" ADD CONSTRAINT "fk_card_holder_id" FOREIGN KEY ("card_holder_id") REFERENCES "card_holder" ("id");
ALTER TABLE "merchant" ADD CONSTRAINT "fk_merchant_category_id" FOREIGN KEY ("merchant_category_id") REFERENCES "merchant_category" ("id");
ALTER TABLE "transaction" ADD CONSTRAINT "fk_card" FOREIGN KEY ("card") REFERENCES "credit_card" ("card");
ALTER TABLE "transaction" ADD CONSTRAINT "fk_id_merchant" FOREIGN KEY ("id_merchant") REFERENCES "merchant" ("id");

