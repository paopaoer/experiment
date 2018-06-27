package shu.ces.company.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.BeanPropertyRowMapper;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import shu.ces.company.model.User;

@Service
public class RegisterService {
    @Autowired
    private JdbcTemplate jdbcTemplate;

    public int add(User user){

        return jdbcTemplate.update("insert into user(user_name,user_password,email,phone_number,head_portrait_path) values(?,?,?,?,?)",
                user.getUser_name(),user.getUser_password(),user.getEmail(),user.getPhone_number(),user.getHead_portrait_path());
    }


    public int queryCount(){
        String sql="select count(*) from user";
        BeanPropertyRowMapper<User> userBeanPropertyRowMapper =new BeanPropertyRowMapper<>(User.class);
        int count=jdbcTemplate.queryForObject(sql,Integer.class);
        return count;
    }
}
