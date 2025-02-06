import sqlite3 from 'sqlite3'
import {type FastifyInstance} from "fastify";
import {db} from "./db.js";

/**
 * @param {FastifyInstance} fastify
 * @param {Object} options
 */
async function dbConnector (fastify: FastifyInstance) {
    fastify.get('/greg', async (req, res) => {
        try {
            const rows = await new Promise((resolve, reject) => {
                db.all("SELECT rowid AS id, info FROM lorem", (err, rows) => {
                    if (err)
                        reject(err);
                    else
                        resolve(rows);
                });
            });

            res.send(JSON.stringify(rows));
        } catch (error) {
            res.status(500).send("Database error");
        }
    });
    // fastify.register(fastifyMongo, {
    //     url: 'mongodb://localhost:27017/test_database'
    // })
}

// Wrapping a plugin function with fastify-plugin exposes the decorators
// and hooks, declared inside the plugin to the parent scope.
export default dbConnector
