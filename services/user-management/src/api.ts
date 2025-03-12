import fastify from 'fastify'
import {addUser} from "./addUser.js";


const app = fastify();

app.post('/sign-up', addUser);

app.listen({port: 5000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})