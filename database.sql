CREATE TABLE IF NOT EXISTS `Live` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `origin_url` varchar(255) NOT NULL,
    `cdn_url` varchar(255) NOT NULL,
    `live_name` varchar(255),
    `video_path` varchar(255),
    `ocr_path` varchar(255),
    `asr_path` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `CommonBanWord` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `word` varchar(255) NOT NULL,
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `BanSale_RestrictSale` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `thesaurus_type` varchar(255),
    `category` varchar(255),
    `department` varchar(255),
    `keyword` varchar(255),
    `legal_proof` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `HealthProducts` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `filing_num` varchar(255),
    `product_name` varchar(255),
    `filing_company` varchar(255),
    `company_addr` varchar(255),
    `filing_record` varchar(255),
    `health_function` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `DomesticDrug` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `approval_num` varchar(255),
    `standard_code` varchar(255),
    `product_name` varchar(255),
    `dosage_form` varchar(255),
    `product_unit` varchar(255),
    `product_category` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `ImportedDrug` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `registration_num` varchar(255),
    `standard_code` varchar(255),
    `product_name` varchar(255),
    `dosage_form` varchar(255),
    `manufacturer` varchar(255),
    `product_category` varchar(255),
    `country_region` varchar(255),
    `certificate_date` varchar(255),
    `expiration_date` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `DomesticCosmetic` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `registration_num` varchar(255),
    `product_name_ch` varchar(255),
    `category` varchar(255),
    `registrant_ch` varchar(255),
    `country_region` varchar(255),
    `approval_date` varchar(255),
    `expiration_date` varchar(255),
    `status` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `ImportedCosmetic` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `registration_num` varchar(255),
    `product_name_ch` varchar(255),
    `category` varchar(255),
    `registrant_ch` varchar(255),
    `country_region` varchar(255),
    `domestic_responsible_person` varchar(255),
    `approval_date` varchar(255),
    `expiration_date` varchar(255),
    `status` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `DomesticMedicalDevice_Registered` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `product_name` varchar(255),
    `management_category` varchar(255),
    `specifications_models` varchar(255),
    `registrant_name` varchar(255),
    `structure_composition` varchar(255),
    `main_components` varchar(255),
    `application_scope` varchar(255),
    `expected_use` varchar(255),
    `approval date` varchar(255),
    `effective date` varchar(255),
    `expiration_date` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `ImportedMedicalDevice_Registered` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `product_name` varchar(255),
    `management_category` varchar(255),
    `specifications_models` varchar(255),
    `registrant_name` varchar(255),
    `agent_name` varchar(255),
    `structure_composition` varchar(255),
    `main_components` varchar(255),
    `application_scope` varchar(255),
    `expected_use` varchar(255),
    `approval date` varchar(255),
    `effective date` varchar(255),
    `expiration_date` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `DomesticMedicalDevice_filing` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `product_name` varchar(255),
    `classification_name` varchar(255),
    `model_specification` varchar(255),
    `packaging_specification` varchar(255),
    `filing_person_Name` varchar(255),
    `product_description` varchar(255),
    `main_components` varchar(255),
    `intended_use` varchar(255),
    `filing_date` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `ImportedMedicalDevice_filing` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `product_name` varchar(255),
    `classification_name` varchar(255),
    `model_specification` varchar(255),
    `packaging_specification` varchar(255),
    `filing_person_Name` varchar(255),
    `agent_name` varchar(255),
    `product_description` varchar(255),
    `main_components` varchar(255),
    `intended_use` varchar(255),
    `filing_date` varchar(255),
    PRIMARY KEY(`ID`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `Product_Match` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `Live_ID` int NOT NULL,
    `Product_ID` int,
    `is_conflict` boolean,
    `specific_conflict` varchar(255),
    PRIMARY KEY(`ID`),
    FOREIGN KEY(`Live_ID`) REFERENCES Live(ID)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `CommonBanWord_Match` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `Live_ID` int NOT NULL,
    `CommonBanWord_ID` int NOT NULL,
    PRIMARY KEY(`ID`),
    FOREIGN KEY(`Live_ID`) REFERENCES Live(ID),
    FOREIGN KEY(`CommonBanWord_ID`) REFERENCES CommonBanWord(ID)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `BanSale_RestrictSale_Match` 
(
    `ID` int NOT NULL AUTO_INCREMENT,
    `Live_ID` int NOT NULL,
    `BanSale_RestrictSale_ID` int NOT NULL,
    PRIMARY KEY(`ID`),
    FOREIGN KEY(`Live_ID`) REFERENCES Live(ID),
    FOREIGN KEY(`BanSale_RestrictSale_ID`) REFERENCES BanSale_RestrictSale(ID)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;
