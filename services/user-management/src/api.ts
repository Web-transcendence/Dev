import fastify from 'fastify'
import {User} from "./User.js";
import {checkToken, checkUser} from "./CheckToken.js";
import {z} from "zod";
import sanitizeHtml from "sanitize-html";
import jwt from "jsonwebtoken";




const clientSchema = z.object({
    name: z.string().min(3, "Minimum 3 caracteres"),
    email: z.string().email("Invalid email"),
    password: z.string().min(6, "Minimum 6 caracteres"),
});



const app = fastify();

app.post('/sign-up', async (req, res) => {
    try {
        const zod_result = clientSchema.safeParse(req.body);
        if (!zod_result.success)
            return res.status(400).send({json: zod_result.error.format()});
        let {name, email, password} = {name: sanitizeHtml(zod_result.data.name), email: sanitizeHtml(zod_result.data.email), password: sanitizeHtml(zod_result.data.password)};
        console.log("data: ", name, email, password);
        if (!name || !email || !password)
            return res.status(454).send({error: "All information are required !"});

        const client: User = new User(name);

        const addRes = await client.addClient(email, password);
        if (!addRes.success) {
            return res.status(409).send({error: `${addRes.result} already exists`});
        }
        return res.status(201).send({token: addRes.result, username: name , redirect: "post/login"});
    }
    catch(err) {
        return res.status(500).send({error: "Server error: ", err});
    }
});

app.post('/check-token', async (req, res) => {
    const {token} = req.body as { token: string };
    if (!token) {
        return res.status(401).send({valid: false, message: "Token is required"});
    }
    const decodedToken = User.verifyToken(token);
    if (!decodedToken) {
        return res.status(401).send({valid: false, message: "Token verification failed"});
    }
    decodedToken.
    return res.status(200).send({token: decodedToken});
    // const user =  new User(decodedToken);
});

app.post('/user-login', checkUser);


app.listen({port: 5000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})