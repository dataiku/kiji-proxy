-- PostgreSQL setup script for Yaak Proxy Service
-- Run this script to create the database and user

-- Database is already created by POSTGRES_DB environment variable
-- CREATE DATABASE pii_proxy;

-- Create user (optional - you can use existing postgres user)
-- CREATE USER pii_proxy_user WITH PASSWORD 'your_secure_password';

-- Grant privileges
-- GRANT ALL PRIVILEGES ON DATABASE pii_proxy TO pii_proxy_user;

-- Already connected to pii_proxy database via POSTGRES_DB

-- Create the pii_mappings table
CREATE TABLE IF NOT EXISTS pii_mappings (
    id SERIAL PRIMARY KEY,
    original_pii VARCHAR(500) NOT NULL UNIQUE,
    dummy_pii VARCHAR(500) NOT NULL UNIQUE,
    pii_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pii_mappings_original ON pii_mappings(original_pii);
CREATE INDEX IF NOT EXISTS idx_pii_mappings_dummy ON pii_mappings(dummy_pii);
CREATE INDEX IF NOT EXISTS idx_pii_mappings_created_at ON pii_mappings(created_at);
CREATE INDEX IF NOT EXISTS idx_pii_mappings_pii_type ON pii_mappings(pii_type);
CREATE INDEX IF NOT EXISTS idx_pii_mappings_last_accessed ON pii_mappings(last_accessed_at);

-- Create a function to automatically update last_accessed_at
CREATE OR REPLACE FUNCTION update_last_accessed()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_accessed_at = NOW();
    NEW.access_count = OLD.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update access statistics
CREATE TRIGGER update_access_stats
    BEFORE UPDATE ON pii_mappings
    FOR EACH ROW
    EXECUTE FUNCTION update_last_accessed();

-- Grant privileges to user (if using dedicated user)
-- GRANT ALL PRIVILEGES ON TABLE pii_mappings TO pii_proxy_user;
-- GRANT ALL PRIVILEGES ON SEQUENCE pii_mappings_id_seq TO pii_proxy_user;

-- Example queries for monitoring and maintenance

-- View all mappings
-- SELECT * FROM pii_mappings ORDER BY created_at DESC LIMIT 10;

-- View most accessed mappings
-- SELECT original_pii, dummy_pii, access_count, last_accessed_at
-- FROM pii_mappings
-- ORDER BY access_count DESC LIMIT 10;

-- View mappings by type
-- SELECT pii_type, COUNT(*) as count
-- FROM pii_mappings
-- GROUP BY pii_type;

-- Cleanup old mappings (older than 7 days)
-- DELETE FROM pii_mappings
-- WHERE created_at < NOW() - INTERVAL '7 days';

-- View database size
-- SELECT pg_size_pretty(pg_database_size('pii_proxy'));

-- View table size
-- SELECT pg_size_pretty(pg_total_relation_size('pii_mappings'));
