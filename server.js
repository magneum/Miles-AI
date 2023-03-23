const apiKey = "860b08c7c72841e6a8a53a3c3cfa8ec6";
import express from "express";
import axios from "axios";
import cors from "cors";
const app = express();
const port = 3000;

app.use(express.json());
app.use(cors()); // add CORS middleware

app.get("/news", async (req, res) => {
  const response = await axios.get(
    "https://newsapi.org/v2/top-headlines?country=in&apiKey=" + apiKey
  );
  console.log(response.data.articles[0]);
  res.send(response.data.articles[0]);
  console.log("Completed...");
});

// app.get("/api/users", (req, res) => {
// // handle GET request for fetching all users
// res.send("List of users");
// });

// app.get("/api/users/:id", (req, res) => {
// // handle GET request for fetching user by id
// const userId = req.params.id;
// res.send(`User with id ${userId}`);
// });

// app.post("/api/users", (req, res) => {
// // handle POST request for creating a new user
// const user = req.body;
// console.log(user);
// res.send("User created successfully");
// });

// app.put("/api/users/:id", (req, res) => {
// // handle PUT request for updating user by id
// const userId = req.params.id;
// const updatedUser = req.body;
// console.log(`Update user with id ${userId}:`, updatedUser);
// res.send(`User with id ${userId} updated successfully`);
// });

// app.delete("/api/users/:id", (req, res) => {
// // handle DELETE request for deleting user by id
// const userId = req.params.id;
// console.log(`Delete user with id ${userId}`);
// res.send(`User with id ${userId} deleted successfully`);
// });

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
